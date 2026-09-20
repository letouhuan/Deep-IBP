#include "Bias.h"
#include "core/ActionRegister.h"
#include "core/ActionSet.h"
#include "core/PlumedMain.h"
#include "core/Atoms.h"
#include "tools/Grid.h"
#include "tools/Communicator.h"
#include "tools/OpenMP.h"
#include "core/FlexibleBin.h"
#include "tools/Exception.h"
#include "tools/Matrix.h"
#include "core/Value.h"
#include "tools/Random.h"
#include "tools/File.h"
#include <Eigen/Dense>
#include <torch/torch.h>
#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using namespace std;

namespace PLMD {
  namespace bias {
    constexpr double kPi = 3.141592653589793238462643383279502884;
    enum Activation { SIGMOID, TANH, RELU, ELU, LINEAR, SWISH, GELU, SOFTPLUS, MISH };
    Activation set_activation(const std::string& a)
    {
      if (a == "SIGMOID" || a == "sigmoid" || a == "Sigmoid")
        return Activation::SIGMOID;
      if (a == "TANH" || a == "tanh" || a == "Tanh")
        return Activation::TANH;
      if (a == "ELU" || a == "elu" || a == "Elu" || a == "eLU")
        return Activation::ELU;
      if (a == "RELU" || a == "relu" || a == "Relu" || a == "ReLU")
        return Activation::RELU;
      if (a == "LINEAR" || a == "linear" || a == "Linear")
        return Activation::LINEAR;
      if (a == "SWISH" || a == "swish" || a == "Swish")
        return Activation::SWISH;
      if (a == "GELU" || a == "gelu" || a == "Gelu")
        return Activation::GELU;
      if (a == "SOFTPLUS" || a == "softplus" || a == "Softplus")
        return Activation::SOFTPLUS;
      if (a == "MISH" || a == "mish" || a == "Mish")
        return Activation::MISH;
      throw std::invalid_argument("Unknown activation function: " + a);
    }
    inline torch::Tensor activate(torch::Tensor x, torch::nn::Linear l, Activation f)
    {
      auto z = l->forward(x);
      switch (f) {
      case LINEAR:
        return z;
      case ELU:
        return torch::elu(z);
      case RELU:
        return torch::relu(z);
      case SIGMOID:
        return torch::sigmoid(z);
      case TANH:
        return torch::tanh(z);
      case SWISH:
        return z * torch::sigmoid(z);
      case GELU:
        return torch::gelu(z);
      case SOFTPLUS:
        return torch::softplus(z);
      case MISH:
        return z * torch::tanh(torch::softplus(z));
      default:
        throw std::invalid_argument("Unknown activation function");
      }
    }
    struct Net : torch::nn::Module {
      Net(std::vector<int> nodes, std::vector<bool> periodic, std::string activ,
        float alpha = 0.0, float beta = 1.0) :
        _layers(),
        _criterion(torch::nn::MSELossOptions().reduction(torch::kNone)),
        _alpha(alpha),
        _beta(beta) {
        if (nodes.size() < 2) {
          throw std::invalid_argument("the neural-network architecture must contain at least input and output layers");
        }
        if (nodes[0] <= 0 || static_cast<std::size_t>(nodes[0]) != periodic.size()) {
          throw std::invalid_argument("the periodicity vector must match the neural-network input dimension");
        }
        _hidden = nodes.size() - 2;
        m_periodic = std::move(periodic);
        nodes[0] = static_cast<int>(std::count(m_periodic.begin(), m_periodic.end(), false)
          + 2 * std::count(m_periodic.begin(), m_periodic.end(), true));
        _activ = set_activation(activ);
        _normalize = false;
        for (int i = 0; i < _hidden; i++) {
          _layers.push_back(register_module("fc" + std::to_string(i + 1), torch::nn::Linear(nodes[i], nodes[i + 1])));
        }
        _out = register_module("out", torch::nn::Linear(nodes[_hidden], nodes[_hidden + 1]));
        register_module("criterion", _criterion);
      }
      void setRange(vector<string> m, vector<string> M) {
        if (m.size() != m_periodic.size() || M.size() != m_periodic.size()) {
          throw std::invalid_argument("normalization ranges must match the input dimension");
        }
        _normalize = true;
        vector<float> mean, inv_range;
        for (unsigned i = 0; i < m.size(); i++) {
          double max, min;
          Tools::convert(m[i], min);
          Tools::convert(M[i], max);
          if (std::abs(max - min) < 1e-8) {
            throw std::invalid_argument(
              "GRID_MAX must differ from GRID_MIN for input dimension " + std::to_string(i));
          }
          mean.push_back((max + min) / 2.);
          inv_range.push_back(2.0 / (max - min));
        }
        _mean = torch::tensor(mean).view({ 1, static_cast<int64_t>(m.size()) });
        _inv_range = torch::tensor(inv_range).view({ 1, static_cast<int64_t>(m.size()) });
      }
      torch::Tensor buildInputFeatures(const torch::Tensor& input) const {
        if (input.dim() != 2 || input.size(1) != static_cast<int64_t>(m_periodic.size())) {
          throw std::invalid_argument("network input must have shape [batch, number_of_CVs]");
        }
        std::vector<torch::Tensor> features;
        features.reserve(2 * m_periodic.size());
        for (std::size_t i = 0; i < m_periodic.size(); ++i) {
          torch::Tensor column = input.slice(1, static_cast<int64_t>(i), static_cast<int64_t>(i + 1));
          if (m_periodic[i]) {
            features.push_back(torch::sin(column));
            features.push_back(torch::cos(column));
          }
          else if (_normalize) {
            const int64_t index = static_cast<int64_t>(i);
            features.push_back((column - _mean.slice(1, index, index + 1))
              * _inv_range.slice(1, index, index + 1));
          }
          else {
            features.push_back(column);
          }
        }
        return torch::cat(features, 1);
      }
      torch::Tensor forward(torch::Tensor x) {
        x = buildInputFeatures(x);
        for (unsigned i = 0; i < _layers.size(); i++)
          x = activate(x, _layers[i], _activ);
        return _out->forward(x);
      }
      torch::Tensor loss(torch::Tensor prediction, torch::Tensor target) {
        torch::Tensor element_wise_mse = _criterion(prediction, target);
        torch::Tensor weight = torch::pow(1.0 + _alpha * torch::abs(target), _beta);
        torch::Tensor weighted_mse = (element_wise_mse * weight).mean();
        return weighted_mse;
      }
      int _hidden;
      bool _normalize;
      vector<bool> m_periodic;
      torch::Tensor _mean, _inv_range;
      vector<torch::nn::Linear> _layers;
      torch::nn::Linear _out{ nullptr };
      Activation _activ;
      torch::nn::MSELoss _criterion;
      float _alpha;
      float _beta;
    };
    class DEEPIBP : public Bias {
    private:
      struct Gaussian {
        bool multivariate;
        double height;
        std::vector<double> center;
        std::vector<double> sigma;
        std::vector<double> invsigma;
        Gaussian(const bool m, const double h, const std::vector<double>& c, const std::vector<double>& s) :
          multivariate(m), height(h), center(c), sigma(s), invsigma(s) {
          for (unsigned i = 0; i < invsigma.size(); ++i) {
            if (std::abs(invsigma[i]) > 1.e-20) invsigma[i] = 1.0 / invsigma[i];
            else invsigma[i] = 0.0;
          }
        }
      };
      unsigned n_sub = 1;
      unsigned num_sub = 0;
      unsigned populated_gaussian = 0;
      unsigned N_unbiased = 0;
      double deltaE = 0.5;
      double height_sub = 0.0;
      std::vector<double> sigma_sub;
      std::vector<std::string> grid_min_str, grid_max_str;
      std::vector<double> height_tall_;
      std::vector<std::vector<double>> sigma_tall_;
      std::vector<double> tall_threshold_;
      std::size_t tall_stage_index_ = 0;
      int tall_check_interval_ = 100;
      bool in_tall_phase_ = false;
      int tall_block_step_count_ = 0;
      int tall_block_inside_count_ = 0;
      std::vector<double> gmin, gmax;
      double biasf_ = -1.0;
      bool welltemp_ = false;
      std::vector<Gaussian> hills_;
      bool guess = false;
      bool subnn = false;
      bool histogram = false;
      bool kl = false;
      bool sumhills_called_ = false;
      bool kde = false;
      bool grid_ = false;
      bool isFirstStep_ = true;
      bool minTOzero = true;
      std::string fmt = "%14.9f";
      std::size_t ncv_ = 0;
      int stride_ = 0;
      int current_stride_ = 0;
      int initial_stride = 0;
      int fes_output_stride_ = 0;
      double temp_ = 0.0;
      std::vector<double> upper_wall;
      std::vector<double> lower_wall;
      std::vector<int> exp_u;
      std::vector<int> exp_l;
      std::vector<double> kappa_u;
      std::vector<double> kappa_l;
      std::unique_ptr<GridBase> BiasGrid_;
      std::string outhills;
      std::string outfiled_fes;
      std::string intergrated_fes;
      std::vector<std::string> ipointname;
      std::vector<std::string> ifilesnames_;
      std::vector<std::unique_ptr<IFile>> ifiles_;
      std::vector<std::unique_ptr<IFile>> pointfile;
      unsigned long hills_count_ = 0;
      unsigned long snapshot_idx_ = 1;
      OFile hillsOfile_;
      bool neutral_network = false;
      bool kl_region = false;
      vector<int> nn_nodes;
      int epochs = 0;
      double loss_alpha = 0.0;
      double loss_beta = 1.0;
      float lrate = 0.001f;
      int num_terms = 0;
      double stretchA = 1.0;
      double stretchB = 0.0;
      std::vector<int> bin_hap;
      int stride_hap = 0;
      std::vector<int> monitor_cvs_;
      std::vector<double> dcv_;
      std::vector<double> bin_tol_;
      std::vector<double> tangential_half_;
      std::vector<double> min_region_;
      double kl_threshold_ = 0.05;
      int samples_min_ = 200;
      int hit_trigger_times_ = 3;
      std::vector<int> upper_hit_count_;
      std::vector<int> lower_hit_count_;
      std::vector<bool> in_upper_bin_;
      std::vector<bool> in_lower_bin_;
      size_t fes_filter_dim_ = 0;
      std::vector<std::vector<double>> cv_trace_;
      bool waiting_unbiased_ = false;
      bool unbiased_phase_entered_ = false;
      unsigned unbiased_phase_start_step_ = 0;
      unsigned unbiased_start_step_ = 0;
      size_t unbiased_start_index_ = 0;
      size_t trigger_dim_for_unbiased_ = 0;
      bool trigger_upper_for_unbiased_ = false;
      std::vector<double> a_box_;
      std::vector<double> b_box_;
      std::vector<double> custom_thr_;
      std::vector<int> custom_op_;
      std::vector<int> custom_hit_count_;
      std::vector<bool> in_custom_bin_;
      vector<bool> periodic;
      shared_ptr<Net> nn_model;
      std::shared_ptr<Net> nn_model_cv1;
      std::shared_ptr<Net> nn_model_cv2;
      std::string optimizer_name_ = "ADAM";
      float beta1_ = 0.9f;
      float beta2_ = 0.999f;
      int kl_gridN_ = 9;
      std::vector<int> kl_row_ok_;
      std::vector<int> kl_col_ok_;
      std::vector<double> kl_row_L_;
      std::vector<double> kl_row_R_;
      std::vector<double> kl_col_low_;
      std::vector<double> kl_col_high_;
      int point_grid = 0;
      std::vector<double> delta_raw_;
      std::vector<double> delta_clamped_;
      std::vector<double> deltaN_over_N_;
      std::vector<bool> delta_frozen_;
      unsigned long small_N_for_dynK_ = 0;
      bool dyn_anchor_initialized_ = false;
      std::vector<double> dyn_anchor_values_;
      bool tall_used_ = false;
      int dynamic_K_ = 0;
      int min_K_ = 8;
      int steep_dim_for_K_ = -1;
      bool dynamic_K_fixed_ = false;
      double dynamic_x_fixed_ = 0.0;
      bool use_dynamic_K_ = false;
      std::vector< std::vector<double> > hill_cv_at_add_;
      static inline double wall_thr(double wall, double d) {
        return wall + d;
      }
      static inline bool hit_with_sign(double cv, double wall, double d) {
        const double t = wall_thr(wall, d);
        return (d >= 0.0) ? (cv >= t) : (cv <= t);
      }
      bool unbiased_anchor_fixed_ = false;
      std::shared_ptr<torch::optim::Optimizer> makeOptimizer(const std::shared_ptr<Net>& model) const {
        if (optimizer_name_ == "SGD") {
          return std::make_shared<torch::optim::SGD>(model->parameters(), torch::optim::SGDOptions(lrate));
        }
        if (optimizer_name_ == "NESTEROV") {
          auto options = torch::optim::SGDOptions(lrate).momentum(0.9).nesterov(true);
          return std::make_shared<torch::optim::SGD>(model->parameters(), options);
        }
        if (optimizer_name_ == "RMSPROP") {
          return std::make_shared<torch::optim::RMSprop>(model->parameters(), torch::optim::RMSpropOptions(lrate));
        }
        if (optimizer_name_ == "ADAGRAD") {
          return std::make_shared<torch::optim::Adagrad>(model->parameters(), torch::optim::AdagradOptions(lrate));
        }
        auto options = torch::optim::AdamOptions(lrate).betas(std::make_tuple(beta1_, beta2_));
        if (optimizer_name_ == "AMSGRAD") options.amsgrad(true);
        return std::make_shared<torch::optim::Adam>(model->parameters(), options);
      }
      inline double custom_anchor_to_thr(double anchor, int op, double dcv) {
        const double w = std::fabs(dcv);
        return (op >= 0) ? (anchor + w) : (anchor - w);
      }
      inline bool hit_custom_anchor(double cv, double anchor, int op, double dcv, int arg_index) {
        const double thr = custom_anchor_to_thr(anchor, op, dcv);
        if (arg_index < 0 ||
          arg_index >= static_cast<int>(periodic.size()) ||
          !periodic[arg_index]) {
          return (op >= 0) ? (cv >= thr) : (cv <= thr);
        }
        const double period_len = gmax[arg_index] - gmin[arg_index];
        if (period_len <= 0.0) {
          return (op >= 0) ? (cv >= thr) : (cv <= thr);
        }
        const double w = std::fabs(dcv);
        const double diff = shortest_periodic_diff(cv, thr, period_len);
        if (op >= 0) {
          return (diff >= 0.0 && diff <= w);
        }
        else {
          return (diff <= 0.0 && diff >= -w);
        }
      }
      inline void custom_range_from_anchor(double anchor, int op, double dcv, double& a, double& b) {
        const double w = std::fabs(dcv);
        if (op >= 0) { a = anchor; b = anchor + w; }
        else { a = anchor - w; b = anchor; }
        if (a > b) std::swap(a, b);
      }
      inline double shortest_periodic_diff(double x, double ref, double period) const {
        double d = x - ref;
        d -= std::round(d / period) * period;
        return d;
      }
      double anchor_distance(int arg_index, double cv_val, double anchor) const {
        if (arg_index < 0 || arg_index >= (int)periodic.size() || !periodic[arg_index]) {
          return std::fabs(cv_val - anchor);
        }
        double period_len = gmax[arg_index] - gmin[arg_index];
        if (period_len <= 0.0) return std::fabs(cv_val - anchor);
        double d = shortest_periodic_diff(cv_val, anchor, period_len);
        return std::fabs(d);
      }
      void addGaussian(const Gaussian& hill);
      std::vector<double> integrateFromDerivatives(const std::vector<double>& cv1, const std::vector<double>& cv2, const std::vector<double>& der_cv1, const std::vector<double>& der_cv2);
      void readBackTwoDFourier(const std::string& filename, int nx, int ny, double x_min, double dx, double y_min, double dy, bool phi_is_x, std::vector<double>& dst);
      double gaussian_kde(const std::vector<double>& data, double x, double h);
      bool scanOneHill(IFile* ifile, std::vector<Value>& tmpvalues, std::vector<double>& center, std::vector<double>& sigma, double& height, bool& multivariate);
      bool scanOnePoly(IFile* ifile, std::vector<Value>& tmpvalues, std::vector<double>& cv, double& bias, std::vector<double>& der);
      void readPoly(IFile* ifile);
      void Addpoly(double bias, std::vector<double> point, std::vector<double> der);
      void writeFESSnapshot_(const std::string& fname, bool restore_after);
      void handleIntervalAndMaybeStop(size_t dim, bool upper_side);
      void writeFESSegmentGridWallCorrected(double a, double b, const std::string& tag);
      bool collectBandSamples2D_Y(double x_min, double x_max,
        double y_min, double y_max,
        std::vector<double>& ys)const;
      double normalized_kde(const std::vector<double>& data, double x, double h, double phi_min, double phi_max, int n_points = 200);
      void runNeuralNetworkTrainingAndOutput();
      double silverman_bandwidth(const std::vector<double>& data);
      void fitOneDimFourier(IFile& infile, OFile& outfile, int n_terms, double xmin_raw, double xmax_raw, int N_grid);
      double gaussian_kde_2d(const std::vector<std::array<double, 2>>& data,
        double x, double y, double hx, double hy);
      std::array<double, 2> silverman_bandwidth_2d(const std::vector<std::array<double, 2>>& data);
      double compute_kl_2d(const std::vector<std::array<double, 2>>& data, const std::array<double, 2>& mins, const std::array<double, 2>& maxs, const std::string& mode, int n_points);
      double computeKLwithKDE(const std::vector<std::vector<double>>& samples, const std::vector<double>& mins, const std::vector<double>& maxs, const std::string& mode, int n_points);
      double getBiasAndDerivatives(const std::vector<double>& cv, std::vector<double>& der);
      void processCV(IFile& ifilecv);
      void updateDynamicKStatistics(const std::vector<double>& cv,unsigned long N,bool this_is_tall);
      void writeGaussian(const Gaussian& hill, OFile& file);
      void twodfourier(IFile& infile, OFile& outfile, int n_terms);
      bool scanonecv(IFile* ifile, std::vector<Value>& tmpvalues, double& time, std::vector<double>& cv);
      std::vector<unsigned> getGaussianSupport(const Gaussian& hill);
      void readGaussians(IFile* ifile);
      double compute_kl(const std::vector<double>& data, double phi_min, double phi_max, const std::string& mode, int n_points = 200);
      double compute_kl_row(const std::vector<double>& data, double phi_min, double phi_max, const std::string& mode, int n_points = 200);
      void sumhills();
      std::vector<double> savgolFilter(const std::vector<double>& y, int window, int polyorder);
      void computeKLFromFile(const std::string& cvfile, double phi_min, double phi_max, unsigned int cv_index);
      std::vector<int> computeHistogram(const std::vector<double>& data, int numBins, double& minVal, double& maxVal, double& binWidth);
      double evaluateGaussianAndDerivatives(const std::vector<double>& cv, const Gaussian& hill, std::vector<double>& der, std::vector<double>& dp_);
      void buildBoxFromCustomThr();
      bool collectBandSamples2D(double x_min, double x_max,
        double y_min, double y_max,
        std::vector<double>& xs) const;
      void computeKLRowsFromCustomThr();
      void computeKLColsFromCustomThr();
      void readcv(IFile& ifile);
      int countHitsForDim(int dim, double anchor, int op, double D) const;
      void buildBoxFromCustomAnchorAndD(double D0, double D1);
      void adjustKAndDWhenKNotReached2D(double& D0, double& D1, int& K_used);
      void writeKL2DSelectedPointsToFiles();
    public:
      explicit DEEPIBP(const ActionOptions&);
      void calculate() override;
      static void registerKeywords(Keywords& keys);
      void update() override;
    };
    PLUMED_REGISTER_ACTION(DEEPIBP, "DEEPIBP")
      void DEEPIBP::registerKeywords(Keywords& keys) {
      Bias::registerKeywords(keys);
      keys.use("ARG");
      keys.add("compulsory", "SIGMA_SUB", "the widths of the Gaussian hills");
      keys.add("compulsory", "HEIGHT_SUB", "height of the small Gaussian hills");
      keys.add("compulsory", "PACE", "the frequency for hill addition");
      keys.add("optional", "N_sub", "the numbers of subspaces");
      keys.add("optional", "NGauss", "Number of filled Gaussians per sub-phase");
      keys.add("optional", "unbSteps", "Unbiased simulation steps");
      keys.add("optional", "deltaE", "Unbiased simulation energy threshold");
      keys.add("optional", "FILE", "file used to read and write Gaussian hills");
      keys.add("optional", "N_Fourier_series", "N_Fourier series");
      keys.add("optional", "GRID_MIN", "the lower bounds for the grid");
      keys.add("optional", "GRID_MAX", "the upper bounds for the grid");
      keys.add("optional", "GRID_BIN", "the number of bins for the grid");
      keys.addFlag("GRID_NOSPLINE", false, "don't use spline interpolation with grids");
      keys.add("optional", "TEMP", "temperature of the simulation");
      keys.addFlag("NN", false, "train and evaluate the neural-network free-energy model");
      keys.add("optional", "GRID_RFILE", "a grid file from which the bias should be read at the initial step of the simulation");
      keys.add("optional", "GRID_WFILE", "the file on which to write the grid");
      keys.addFlag("GUESS", false, "Set to TRUE if you want to compute the metadynamics acceleration factor.");
      keys.add("optional", "GUESS_RFILE", "a data file from which the acceleration should be read at the initial step of the simulation");
      keys.addFlag("SUBNN", false, "subnn");
      keys.add("optional", "OPTIM", "choose the optimizer");
      keys.add("optional", "EPOCH", "epoch");
      keys.add("optional", "EXP_U", "upper_wall_EXP");
      keys.add("optional", "EXP_L", "upper_wall_EXP");
      keys.add("optional", "KAPPA_U", "upper_wall_KAPPA");
      keys.add("optional", "KAPPA_L", "upper_wall_KAPPA");
      keys.add("optional", "UPPER_WALL", "upper_wall");
      keys.add("optional", "GRID_SPACING", "GRID_SPACING");
      keys.addFlag("KL", false, "use kl");
      keys.add("optional", "LOWER_WALL", "lower_wall");
      keys.add("optional", "BETA1", "b1 coeff of ADAM");
      keys.add("optional", "BIASFACTOR", "use well tempered metadynamics and use this bias factor.  Please note you must also specify temp");
      keys.add("optional", "BETA2", "b2 coeff of ADAM");
      keys.add("optional", "ACTIVATION", "activation function for hidden layers");
      keys.add("optional", "NODES", "neural network architecture");
      keys.add("optional", "LRATE", "the step used for the minimization of the functional");
      keys.add("optional", "STRIDEFES", "Interval steps for outputting free energy");
      keys.add("optional", "BINFES", "bin of fes.dat");
      keys.addFlag("GRID_SPARSE", false, "use a sparse grid to store hills");
      keys.addFlag("mintozero", false, "mintozero");
      keys.addFlag("HISTOGRAMfes", false, "Using histogram-based local convergence criteria");
      keys.addFlag("KDE", false, "Using KDE-based local convergence criteria");
      keys.add("optional", "FES_OUTPUT", "output one FES file every N steps using HILLS");
      keys.addFlag("KL_REGION", false, "UsingKL_REGION");
      keys.add("optional", "DCV", "Offsets from wall (vector, one per monitored CV)");
      keys.add("optional", "BIN_TOL", "Normal band half-widths (vector, default auto)");
      keys.add("optional", "TANGENTIAL_HALF", "Tangential half-widths (vector, default 2*sigma)");
      keys.add("optional", "MIN_REGION", "Minimum interval length (vector, one per CV)");
      keys.add("optional", "MONITOR_CVS", "Indices of CVs to monitor (vector, default first N CVs)");
      keys.add("optional", "KL_THRESHOLD", "KL divergence threshold");
      keys.add("optional", "SAMPLES_MIN", "Minimum samples required for KL check");
      keys.add("optional", "HIT_TRIGGER", "Number of hits to trigger unbiased window (default 3)");
      keys.add("optional", "CUSTOM_THR", "User-defined threshold per CV (used only if both walls present or none)");
      keys.add("optional", "CUSTOM_THR_OP", "Operator per CV: GE or LE (>= or <=); default GE");
      keys.add("optional", "POINT_GRID", "resolution of the output POINT grid");
      keys.add("optional", "SIGMA_TALL", "widths of the staged tall Gaussian hills; for multiple stages and multiple CVs use stage-major order");
      keys.add("optional", "HEIGHT_TALL", "heights of the staged tall Gaussian hills");
      keys.addFlag("ADAPTIVE_K", false, "Use adaptive K to update HIT_TRIGGER dynamically");
      keys.add("optional", "TALL_CHECK_INTERVAL", "interval (in MD steps) to evaluate tall-packet occupancy");
      keys.add("optional", "TALL_THRESHOLD", "one occupancy threshold per tall stage; each threshold advances to the next tall stage, and the last switches to small packets");
      keys.add("optional", "LOSS_ALPHA", "scale factor in the derivative-weighted MSE loss");
      keys.add("optional", "LOSS_BETA", "exponent in the derivative-weighted MSE loss");
    }
    DEEPIBP::DEEPIBP(const ActionOptions& ao) :
      PLUMED_BIAS_INIT(ao) {
      parse("LOSS_ALPHA", loss_alpha);
      parse("LOSS_BETA", loss_beta);
      parseVector("SIGMA_SUB", sigma_sub);
      if (sigma_sub.size() != getNumberOfArguments())
      {
        error("number of arguments does not match number of SIGMA parameters");
      }
      if (std::any_of(sigma_sub.begin(), sigma_sub.end(),
        [](double sigma) { return !std::isfinite(sigma) || sigma <= 0.0; })) {
        error("all SIGMA_SUB values must be finite and positive");
      }
      parse("HEIGHT_SUB", height_sub);
      std::vector<double> sigma_tall_flat;
      parseVector("SIGMA_TALL", sigma_tall_flat);
      parseVector("HEIGHT_TALL", height_tall_);
      parse("TALL_CHECK_INTERVAL", tall_check_interval_);
      parseVector("TALL_THRESHOLD", tall_threshold_);
      const std::size_t ncv_for_tall = getNumberOfArguments();
      if (height_tall_.empty()) {
        height_tall_.assign(1, height_sub);
      }
      for (std::size_t stage = 0; stage < height_tall_.size(); ++stage) {
        if (height_tall_[stage] <= 0.0) height_tall_[stage] = height_sub;
      }
      const std::size_t n_tall_stages = height_tall_.size();
      sigma_tall_.assign(n_tall_stages, sigma_sub);
      if (!sigma_tall_flat.empty()) {
        if (n_tall_stages == 1 && sigma_tall_flat.size() == ncv_for_tall) {
          sigma_tall_[0] = sigma_tall_flat;
        }
        else if (sigma_tall_flat.size() == n_tall_stages * ncv_for_tall) {
          for (std::size_t stage = 0; stage < n_tall_stages; ++stage) {
            for (std::size_t d = 0; d < ncv_for_tall; ++d) {
              sigma_tall_[stage][d] = sigma_tall_flat[stage * ncv_for_tall + d];
            }
          }
        }
        else {
          std::ostringstream oss;
          oss << "SIGMA_TALL must contain " << ncv_for_tall
              << " value(s) for one tall stage, or "
              << (n_tall_stages * ncv_for_tall)
              << " values for " << n_tall_stages
              << " tall stages in stage-major order";
          error(oss.str());
        }
      }
      for (std::size_t stage = 0; stage < sigma_tall_.size(); ++stage) {
        for (std::size_t d = 0; d < sigma_tall_[stage].size(); ++d) {
          if (sigma_tall_[stage][d] <= 0.0) {
            error("all SIGMA_TALL values must be positive");
          }
        }
      }
      if (tall_threshold_.empty()) {
        if (n_tall_stages == 1) tall_threshold_.assign(1, 0.7);
        else error("multiple HEIGHT_TALL values require the same number of TALL_THRESHOLD values");
      }
      if (tall_threshold_.size() != n_tall_stages) {
        error("TALL_THRESHOLD size must equal HEIGHT_TALL size");
      }
      for (std::size_t stage = 0; stage < tall_threshold_.size(); ++stage) {
        if (tall_threshold_[stage] <= 0.0 || tall_threshold_[stage] > 1.0) {
          error("each TALL_THRESHOLD must be in the interval (0,1]");
        }
        if (stage > 0 && tall_threshold_[stage] <= tall_threshold_[stage - 1]) {
          error("TALL_THRESHOLD values must be strictly increasing");
        }
      }
      if (tall_check_interval_ <= 0) tall_check_interval_ = 100;
      tall_stage_index_ = 0;
      parseFlag("GUESS", guess);
      parseFlag("KL", kl);
      parseFlag("KL_REGION", kl_region);
      parseFlag("ADAPTIVE_K", use_dynamic_K_);
      parseFlag("mintozero", minTOzero);
      parse("STRIDEFES", stride_hap);
      parseVector("BINFES", bin_hap);
      ncv_ = getNumberOfArguments();
      std::string guess_rfilename;
      if (guess) {
        log.printf(" The guess mode has been activated.\n");
        parse("GUESS_RFILE", guess_rfilename);
        bool file_parsed = !guess_rfilename.empty();
        if (!file_parsed) {
          log.printf(" ERROR: GUESS mode was enabled but no GUESS_RFILE provided.\n");
          error("GUESS_RFILE must be specified when GUESS is active.");
        }
        log.printf(" GUESS_RFILE set to: %s\n", guess_rfilename.c_str());
      }
      std::string hillsfname = "HILLS";
      parse("FILE", hillsfname);
      parseFlag("HISTOGRAMfes", histogram);
      parseFlag("KDE", kde);
      fes_output_stride_ = 0;
      parse("FES_OUTPUT", fes_output_stride_);
      if (histogram) {
        log.printf(" Using histogram-based local convergence criteria. \n");
      }
      if (kde) {
        log.printf(" Using kde-based local convergence criteria. \n");
      }
      parseFlag("SUBNN", subnn);
      if (subnn) {
        log.printf(" The subnn mode has been activated. \n");
      }
      deltaE = 0.5;
      parse("deltaE", deltaE);
      parse("BIASFACTOR", biasf_);
      if (biasf_ < 1.0 && biasf_ != -1.0)
      {
        error("well tempered bias factor is nonsensical");
      }
      welltemp_ = biasf_ > 1.0;
      parseVector("GRID_MIN", grid_min_str);
      parseVector("GRID_MAX", grid_max_str);
      if (grid_min_str.size() != getNumberOfArguments() && !grid_min_str.empty())
      {
        error("not enough values for GRID_MIN");
      }
      if (grid_max_str.size() != getNumberOfArguments() && !grid_max_str.empty())
      {
        error("not enough values for GRID_MAX");
      }
      if (grid_min_str.size() != grid_max_str.size()) {
        error("GRID_MAX and GRID_MIN should be either present or absent");
      }
      gmin.resize(grid_min_str.size());
      gmax.resize(grid_max_str.size());
      for (size_t i = 0; i < grid_min_str.size(); ++i) {
        Tools::convert(grid_min_str[i], gmin[i]);
        Tools::convert(grid_max_str[i], gmax[i]);
      }
      std::vector<unsigned> gbin(getNumberOfArguments());
      parseVector("GRID_BIN", gbin);
      if (std::any_of(gbin.begin(), gbin.end(), [](unsigned bins) { return bins == 0; })) {
        error("all GRID_BIN values must be positive");
      }
      if (gbin.size() > 0) grid_ = true;
      std::vector<double> gspacing;
      parseVector("GRID_SPACING", gspacing);
      if (gspacing.size() != getNumberOfArguments() && gspacing.size() != 0)
      {
        error("not enough values for GRID_SPACING");
      }
      if (gmin.size() != gmax.size())
      {
        error("GRID_MAX and GRID_MIN should be either present or absent");
      }
      if (gspacing.size() != 0 && gmin.size() == 0)
      {
        error("If GRID_SPACING is present also GRID_MIN and GRID_MAX should be present");
      }
      if (gbin.size() != 0 && gmin.size() == 0)
      {
        error("If GRID_BIN is present also GRID_MIN and GRID_MAX should be present");
      }
      if (gmin.size() != 0) {
        if (gbin.size() == 0 && gspacing.size() == 0) {
          log << "  Binsize not specified, 1/5 of sigma will be be used\n";
          if (guess) {
            plumed_assert(sigma_sub.size() == getNumberOfArguments());
            gspacing.resize(getNumberOfArguments());
            for (unsigned i = 0; i < gspacing.size(); i++) gspacing[i] = 0.2 * sigma_sub[i];
          }
          if (subnn)
          {
            plumed_assert(sigma_sub.size() == getNumberOfArguments());
            gspacing.resize(getNumberOfArguments());
            for (unsigned i = 0; i < gspacing.size(); i++) gspacing[i] = 0.2 * sigma_sub[i];
          }
        }
      }
      bool nospline = false;
      parseFlag("GRID_NOSPLINE", nospline);
      bool spline = !nospline;
      parse("N_Fourier_series", num_terms);
      const unsigned nn_dim = getNumberOfArguments();
      parseFlag("NN", neutral_network);
      parse("POINT_GRID", point_grid);
      if (neutral_network && point_grid < 2) {
        error("POINT_GRID must be >= 2");
      }
      parseVector("NODES", nn_nodes);
      nn_nodes.insert(nn_nodes.begin(), nn_dim);
      nn_nodes.push_back(1);
      lrate = 0.001;
      parse("LRATE", lrate);
      periodic.resize(getNumberOfArguments());
      for (unsigned i = 0; i < getNumberOfArguments(); i++)
        periodic[i] = getPntrToArgument(i)->isPeriodic();
      std::string activation = "ELU";
      parse("ACTIVATION", activation);
      if (neutral_network) {
        nn_model = std::make_shared<Net>(nn_nodes, periodic, activation, loss_alpha, loss_beta);
        nn_model_cv1 = std::make_shared<Net>(nn_nodes, periodic, activation, loss_alpha, loss_beta);
        nn_model_cv2 = std::make_shared<Net>(nn_nodes, periodic, activation, loss_alpha, loss_beta);
      }
      std::vector<std::string> gmin_str(gmin.size()), gmax_str(gmax.size());
      for (size_t i = 0; i < gmin.size(); ++i) {
        gmin_str[i] = std::to_string(gmin[i]);
        gmax_str[i] = std::to_string(gmax[i]);
      }
      if (neutral_network && gmin_str.size() == ncv_ && gmax_str.size() == ncv_) {
        nn_model->setRange(gmin_str, gmax_str);
        nn_model_cv1->setRange(gmin_str, gmax_str);
        nn_model_cv2->setRange(gmin_str, gmax_str);
      }
      std::string opt = "ADAM";
      parse("OPTIM", opt);
      std::transform(opt.begin(), opt.end(), opt.begin(),
        [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
      const std::vector<std::string> valid_optimizers = {
        "SGD", "NESTEROV", "RMSPROP", "ADAM", "ADAGRAD", "AMSGRAD"
      };
      if (std::find(valid_optimizers.begin(), valid_optimizers.end(), opt) == valid_optimizers.end()) {
        error("OPTIM must be one of SGD, NESTEROV, RMSPROP, ADAM, ADAGRAD, or AMSGRAD");
      }
      optimizer_name_ = opt;
      parse("BETA1", beta1_);
      parse("BETA2", beta2_);
      if (beta1_ <= 0.0f || beta1_ >= 1.0f || beta2_ <= 0.0f || beta2_ >= 1.0f) {
        error("BETA1 and BETA2 must lie strictly between zero and one");
      }
      double temp = 0;
      parse("TEMP", temp);
      temp_ = temp;
      double KbT = plumed.getAtoms().getKBoltzmann() * temp;
      if (KbT <= 0) {
        KbT = plumed.getAtoms().getKbT();
        plumed_massert(KbT > 0, "your MD engine does not pass the temperature to plumed, you must specify it using TEMP");
      }
      parse("N_sub", n_sub);
      outhills = "fes_" + std::to_string(n_sub) + ".dat";
      outfiled_fes = "filtered_fes_" + std::to_string(n_sub) + ".dat";
      intergrated_fes = "intergrated_fes.dat";
      parse("PACE", stride_);
      if (stride_ <= 0) error("frequency for hill addition is nonsensical");
      current_stride_ = stride_;
      initial_stride = stride_;
      parse("NGauss", num_sub);
      parse("unbSteps", N_unbiased);
      parse("EPOCH", epochs);
      if (!neutral_network && num_sub == 0) {
        error("NGauss must be positive when Gaussian sampling is active");
      }
      if (neutral_network) {
        if (epochs <= 0) error("EPOCH must be positive when NN is active");
        if (gmin.size() != ncv_ || gmax.size() != ncv_) {
          error("GRID_MIN and GRID_MAX are required when NN is active");
        }
        if (std::any_of(periodic.begin(), periodic.end(), [](bool value) { return value; })
          && num_terms <= 0) {
          error("N_Fourier_series must be positive for periodic NN inputs");
        }
      }
      parseVector("UPPER_WALL", upper_wall);
      parseVector("EXP_U", exp_u);
      parseVector("EXP_L", exp_l);
      parseVector("KAPPA_U", kappa_u);
      parseVector("KAPPA_L", kappa_l);
      parseVector("LOWER_WALL", lower_wall);
      const size_t ncv = getNumberOfArguments();
      auto expand_or_default_double = [this, ncv](std::vector<double>& values,
        double default_value, const char* name) {
          if (values.empty()) values.assign(ncv, default_value);
          else if (values.size() == 1 && ncv > 1) values.resize(ncv, values[0]);
          else if (values.size() != ncv) error(std::string(name) + " must contain one value or one value per CV");
        };
      auto expand_or_default_int = [this, ncv](std::vector<int>& values,
        int default_value, const char* name) {
          if (values.empty()) values.assign(ncv, default_value);
          else if (values.size() == 1 && ncv > 1) values.resize(ncv, values[0]);
          else if (values.size() != ncv) error(std::string(name) + " must contain one value or one value per CV");
        };
      expand_or_default_double(lower_wall, -std::numeric_limits<double>::infinity(), "LOWER_WALL");
      expand_or_default_double(upper_wall, std::numeric_limits<double>::infinity(), "UPPER_WALL");
      expand_or_default_double(kappa_l, 0.0, "KAPPA_L");
      expand_or_default_double(kappa_u, 0.0, "KAPPA_U");
      expand_or_default_int(exp_l, 2, "EXP_L");
      expand_or_default_int(exp_u, 2, "EXP_U");
      std::vector<int> monitor_cvs_vec;
      parseVector("MONITOR_CVS", monitor_cvs_vec);
      if (kl_region && monitor_cvs_vec.empty()) {
        if (getNumberOfArguments() >= 2) monitor_cvs_vec = { 0,1 };
        else monitor_cvs_vec = { 0 };
      }
      monitor_cvs_ = monitor_cvs_vec;
      {
        std::vector<bool> seen(ncv, false);
        for (int index : monitor_cvs_) {
          if (index < 0 || static_cast<std::size_t>(index) >= ncv) {
            error("each MONITOR_CVS index must refer to an existing CV");
          }
          if (seen[static_cast<std::size_t>(index)]) {
            error("MONITOR_CVS must not contain duplicate indices");
          }
          seen[static_cast<std::size_t>(index)] = true;
        }
      }
      parseVector("DCV", dcv_);
      const std::size_t nmon = monitor_cvs_.size();
      delta_raw_.assign(nmon, 0.0);
      delta_clamped_.assign(nmon, 0.0);
      deltaN_over_N_.assign(nmon, 0.0);
      delta_frozen_.assign(nmon, false);
      tall_used_ = false;
      dynamic_K_ = 0;
      min_K_ = 8;
      dynamic_K_fixed_ = false;
      steep_dim_for_K_ = -1;
      if (dcv_.size() == 1 && monitor_cvs_.size() > 1) {
        dcv_.resize(monitor_cvs_.size(), dcv_[0]);
      }
      if (dcv_.size() != monitor_cvs_.size()) error("DCV length must match number of monitored CVs");
      parseVector("BIN_TOL", bin_tol_);
      if (bin_tol_.empty()) {
        bin_tol_.assign(monitor_cvs_.size(), -1.0);
      }
      if (bin_tol_.size() == 1 && monitor_cvs_.size() > 1) {
        bin_tol_.resize(monitor_cvs_.size(), bin_tol_[0]);
      }
      if (bin_tol_.size() != monitor_cvs_.size()) error("BIN_TOL length mismatch");
      parseVector("TANGENTIAL_HALF", tangential_half_);
      if (tangential_half_.empty()) {
        tangential_half_.resize(monitor_cvs_.size());
        for (size_t i = 0; i < monitor_cvs_.size(); ++i)
          tangential_half_[i] = 2.0 * sigma_sub[monitor_cvs_[i]];
      }
      if (tangential_half_.size() == 1 && monitor_cvs_.size() > 1) {
        tangential_half_.resize(monitor_cvs_.size(), tangential_half_[0]);
      }
      if (tangential_half_.size() != monitor_cvs_.size()) error("TANGENTIAL_HALF length mismatch");
      parseVector("MIN_REGION", min_region_);
      if (min_region_.empty()) {
        min_region_.assign(monitor_cvs_.size(), 0.05);
      }
      if (min_region_.size() == 1 && monitor_cvs_.size() > 1) {
        min_region_.resize(monitor_cvs_.size(), min_region_[0]);
      }
      if (min_region_.size() != monitor_cvs_.size()) error("MIN_REGION length mismatch");
      parse("KL_THRESHOLD", kl_threshold_);
      parse("SAMPLES_MIN", samples_min_);
      parse("HIT_TRIGGER", hit_trigger_times_);
      if (hit_trigger_times_ < 1) {
        error("HIT_TRIGGER must be >= 1");
      }
      log.printf("[KL-REGION] ADAPTIVE_K=%d, initial HIT_TRIGGER=%d\n",
        use_dynamic_K_ ? 1 : 0, hit_trigger_times_);
      upper_hit_count_.assign(monitor_cvs_.size(), 0);
      lower_hit_count_.assign(monitor_cvs_.size(), 0);
      in_upper_bin_.assign(monitor_cvs_.size(), false);
      in_lower_bin_.assign(monitor_cvs_.size(), false);
      const size_t ndim = monitor_cvs_.size();
      parseVector("CUSTOM_THR", custom_thr_);
      std::vector<std::string> op_str;
      parseVector("CUSTOM_THR_OP", op_str);
      if (!custom_thr_.empty() && custom_thr_.size() != ndim) {
        if (custom_thr_.size() == 1) custom_thr_.assign(ndim, custom_thr_[0]);
        else error("CUSTOM_THR length must be 1 or equal to number of monitored CVs");
      }
      custom_op_.assign(ndim, +1);
      if (!op_str.empty()) {
        if (op_str.size() == 1) op_str.assign(ndim, op_str[0]);
        else if (op_str.size() != ndim) error("CUSTOM_THR_OP length must be 1 or equal to number of monitored CVs");
        for (size_t k = 0; k < ndim; ++k) {
          std::string s = op_str[k];
          for (char& c : s) {
            c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
          }
          if (s == "GE" || s == ">=") custom_op_[k] = +1;
          else if (s == "LE" || s == "<=") custom_op_[k] = -1;
          else error("CUSTOM_THR_OP accepts GE/LE (or >=/<=)");
        }
      }
      custom_hit_count_.assign(ndim, 0);
      in_custom_bin_.assign(ndim, false);
      tall_stage_index_ = 0;
      in_tall_phase_ = (kl_region &&
        (monitor_cvs_.size() == 1 || monitor_cvs_.size() == 2) &&
        !height_tall_.empty());
      tall_block_step_count_ = 0;
      tall_block_inside_count_ = 0;
      if (in_tall_phase_) {
        log.printf("[TALL-PHASE] configured %zu tall stage(s), check_interval=%d\n",
          height_tall_.size(), tall_check_interval_);
        for (std::size_t stage = 0; stage < height_tall_.size(); ++stage) {
          log.printf("[TALL-PHASE] stage=%zu height=%.6f threshold=%.6f sigma=",
            stage + 1, height_tall_[stage], tall_threshold_[stage]);
          for (std::size_t d = 0; d < sigma_tall_[stage].size(); ++d) {
            log.printf("%s%.6f", (d == 0 ? "" : ","), sigma_tall_[stage][d]);
          }
          log.printf("\n");
        }
      }
      std::string gridfilename_;
      parse("GRID_WFILE", gridfilename_);
      std::string gridreadfilename_;
      parse("GRID_RFILE", gridreadfilename_);
      if (!grid_ && gridfilename_.length() > 0) error("To write a grid you need first to define it!");
      if (!grid_ && gridreadfilename_.length() > 0) error("To read a grid you need first to define it!");
      bool sparsegrid = false;
      parseFlag("GRID_SPARSE", sparsegrid);
      if (grid_) {
        if (!(gridreadfilename_.length() > 0)) {
          for (unsigned i = 0; i < getNumberOfArguments(); i++) {
            double a, b;
            Tools::convert(grid_min_str[i], a);
            Tools::convert(grid_max_str[i], b);
            double mesh = (b - a) / ((double)gbin[i]);
            if (mesh > 0.5 * sigma_sub[i]) log << "  WARNING: Using a METAD with a Grid Spacing larger than half of the Gaussians width (SIGMA) can produce artifacts\n";
          }
          std::string funcl = getLabel() + ".bias";
          if (!sparsegrid) { BiasGrid_ = Tools::make_unique<Grid>(funcl, getArguments(), grid_min_str, grid_max_str, gbin, spline, true); }
          else { BiasGrid_ = Tools::make_unique<SparseGrid>(funcl, getArguments(), grid_min_str, grid_max_str, gbin, spline, true); }
          std::vector<std::string> actualmin = BiasGrid_->getMin();
          std::vector<std::string> actualmax = BiasGrid_->getMax();
          for (unsigned i = 0; i < getNumberOfArguments(); i++) {
            std::string is;
            Tools::convert(i, is);
            if (grid_min_str[i] != actualmin[i]) error("GRID_MIN[" + is + "] must be adjusted to " + actualmin[i] + " to fit periodicity");
            if (grid_max_str[i] != actualmax[i]) error("GRID_MAX[" + is + "] must be adjusted to " + actualmax[i] + " to fit periodicity");
          }
        }
        else {
          IFile gridfile;
          gridfile.link(*this);
          if (gridfile.FileExist(gridreadfilename_)) {
            gridfile.open(gridreadfilename_);
          }
          else {
            error("The GRID file you want to read: " + gridreadfilename_ + ", cannot be found!");
          }
          std::string funcl = getLabel() + ".bias";
          BiasGrid_ = GridBase::create(funcl, getArguments(), gridfile, grid_min_str, grid_max_str, gbin, sparsegrid, spline, true);
          if (BiasGrid_->getDimension() != getNumberOfArguments()) error("mismatch between dimensionality of input grid and number of arguments");
          for (unsigned i = 0; i < getNumberOfArguments(); ++i) {
            if (getPntrToArgument(i)->isPeriodic() != BiasGrid_->getIsPeriodic()[i]) error("periodicity mismatch between arguments and input bias");
            double a, b;
            Tools::convert(grid_min_str[i], a);
            Tools::convert(grid_max_str[i], b);
            double mesh = (b - a) / ((double)gbin[i]);
            if (mesh > 0.5 * sigma_sub[i]) log << "  WARNING: Using a METAD with a Grid Spacing larger than half of the Gaussians width can produce artifacts\n";
          }
          log.printf("  Restarting from %s\n", gridreadfilename_.c_str());
        }
      }
      std::string pofname;
      pofname = guess_rfilename;
      pointfile.emplace_back(Tools::make_unique<IFile>());
      IFile* point_ifile = pointfile.back().get();
      ipointname.push_back(pofname);
      point_ifile->link(*this);
      if (point_ifile->FileExist(pofname)) {
        point_ifile->open(pofname);
        if (guess) {
          log.printf("using point\n");
          readPoly(point_ifile);
          writeFESSnapshot_("hap_0.dat", true);
          snapshot_idx_ = 1;
        }
        point_ifile->reset(false);
        pointfile.back()->close();
      }
      hillsOfile_.link(*this);
      std::string fname;
      fname = hillsfname;
      ifiles_.emplace_back(Tools::make_unique<IFile>());
      IFile* ifile = ifiles_.back().get();
      ifilesnames_.push_back(fname);
      ifile->link(*this);
      if (ifile->FileExist(fname)) {
        ifile->open(fname);
        if (getRestart()) {
          log.printf("  Restarting from %s:", fname.c_str());
          readGaussians(ifile);
        }
        ifile->reset(false);
        ifiles_.back()->close();
      }
      hillsOfile_.open(fname);
      hillsOfile_.addConstantField("multivariate");
      hillsOfile_.addConstantField("kerneltype");
      hillsOfile_.setHeavyFlush();
      for (unsigned i = 0; i < getNumberOfArguments(); ++i) hillsOfile_.setupPrintValue(getPntrToArgument(i));
      if (monitor_cvs_.size() >= 2) {
        const size_t d0 = monitor_cvs_[0];
        const size_t d1 = monitor_cvs_[1];
        if (a_box_.size() < 2) a_box_.assign(2, 0.0);
        if (b_box_.size() < 2) b_box_.assign(2, 0.0);
        a_box_[0] = lower_wall[d0];
        b_box_[0] = upper_wall[d0];
        a_box_[1] = lower_wall[d1];
        b_box_[1] = upper_wall[d1];
      }
      checkRead();
      log.printf("  Gaussian width ");
      if (guess)
      {
        log.printf("  Gaussian width ");
        for (unsigned i = 0; i < sigma_sub.size(); ++i) log.printf(" %f", sigma_sub[i]);
        log.printf("  Gaussian height %f\n", height_sub);
      }
      if (subnn)
      {
        log.printf("  Gaussian width ");
        for (unsigned i = 0; i < sigma_sub.size(); ++i) log.printf(" %f", sigma_sub[i]);
        log.printf("  Gaussian height %f\n", height_sub);
      }
      log.printf("  Gaussian deposition pace %d\n", stride_);
      log.printf("  Gaussian file %s\n", hillsfname.c_str());
      if (grid_) {
        log.printf("GRID_MIN:");
        for (unsigned i = 0; i < gmin.size(); ++i) {
          log.printf(" %.6f", gmin[i]);
        }
        log.printf("\n");
        log.printf("GRID_MAX:");
        for (unsigned i = 0; i < gmax.size(); ++i) {
          log.printf(" %.6f", gmax[i]);
        }
        log.printf("\n");
        log.printf("  Grid bin");
        for (unsigned i = 0; i < gbin.size(); ++i) log.printf(" %u", gbin[i]);
        log.printf("\n");
        if (spline) { log.printf("  Grid uses spline interpolation\n"); }
      }
      log << "  Bibliography " << plumed.cite("Laio and Parrinello, PNAS 99, 12562 (2002)");
      if (welltemp_) log << plumed.cite("Barducci, Bussi, and Parrinello, Phys. Rev. Lett. 100, 020603 (2008)");
      log << "\n";
    }
    void DEEPIBP::Addpoly(double bias, std::vector<double> point, std::vector<double> der) {
      if (grid_) {
        size_t ncv = getNumberOfArguments();
        std::vector<Grid::index_t> index_p;
        index_p.push_back(BiasGrid_->getIndex(point));
        Grid::index_t index_pp;
        for (size_t i = 0; i < index_p.size(); ++i)
        {
          index_pp = index_p[i];
        }
        BiasGrid_->setValueAndDerivatives(index_pp, bias, der);
        log.printf("point %u ,poly %f\n", index_pp, bias);
      }
      else {
        log.printf("nogrid");
      }
    }
    bool DEEPIBP::scanOnePoly(IFile* ifile, std::vector<Value>& tmpvalues, std::vector<double>& cv, double& bias, std::vector<double>& der)
    {
      unsigned ncv = tmpvalues.size();
      if (ifile->scanField("bias", bias)) {
        for (unsigned i = 0; i < ncv; ++i) {
          ifile->scanField(&tmpvalues[i]);
          cv[i] = tmpvalues[i].get();
        }
        for (unsigned i = 0; i < ncv; ++i) {
          ifile->scanField("der_" + getPntrToArgument(i)->getName(), der[i]);
        }
        ifile->scanField();
        return true;
      }
      else {
        return false;
      }
    }
    void DEEPIBP::readPoly(IFile* ifile) {
      unsigned ncv = getNumberOfArguments();
      std::vector<double> point(ncv);
      std::vector<double> der(ncv);
      int npoly = 0;
      double bias = 0.0;
      std::vector<Value> tmpvalues;
      for (unsigned j = 0; j < getNumberOfArguments(); ++j) tmpvalues.push_back(Value(this, getPntrToArgument(j)->getName(), false));
      while (scanOnePoly(ifile, tmpvalues, point, bias, der))
      {
        npoly++;
        Addpoly(bias, point, der);
      }
      log.printf("      %d Point read\n", npoly);
    }
    double DEEPIBP::evaluateGaussianAndDerivatives(const std::vector<double>& cv, const Gaussian& hill, std::vector<double>& der, std::vector<double>& dp_)
    {
      unsigned ncv = cv.size();
      const double* pcv = cv.data();
      double dp2 = 0.0;
      double bias = 0.0;
      if (hill.multivariate) {
        unsigned k = 0;
        Matrix<double> mymatrix(ncv, ncv);
        for (unsigned i = 0; i < ncv; i++) {
          for (unsigned j = i; j < ncv; j++) {
            mymatrix(i, j) = mymatrix(j, i) = hill.sigma[k];
            k++;
          }
        }
        for (unsigned i = 0; i < ncv; i++) {
          dp_[i] = difference(i, hill.center[i], pcv[i]);
          for (unsigned j = i; j < ncv; j++) {
            if (i == j) {
              dp2 += dp_[i] * dp_[i] * mymatrix(i, j) * 0.5;
            }
            else {
              double dp_j = difference(j, hill.center[j], pcv[j]);
              dp2 += dp_[i] * dp_j * mymatrix(i, j);
            }
          }
        }
        if (dp2 < dp2cutoff) {
          bias = hill.height * std::exp(-dp2);
          for (unsigned i = 0; i < ncv; i++) {
            double tmp = 0.0;
            for (unsigned j = 0; j < ncv; j++) tmp += dp_[j] * mymatrix(i, j) * bias;
            der[i] -= tmp * stretchA;
          }
          bias = stretchA * bias + hill.height * stretchB;
        }
      }
      else {
        for (unsigned i = 0; i < ncv; i++) {
          dp_[i] = difference(i, hill.center[i], pcv[i]) * hill.invsigma[i];
          dp2 += dp_[i] * dp_[i];
        }
        dp2 *= 0.5;
        if (dp2 < dp2cutoff) {
          bias = hill.height * std::exp(-dp2);
          for (unsigned i = 0; i < ncv; i++) der[i] -= bias * dp_[i] * hill.invsigma[i] * stretchA;
          bias = stretchA * bias + hill.height * stretchB;
        }
      }
      return bias;
    }
    std::vector<unsigned> DEEPIBP::getGaussianSupport(const Gaussian& hill)
    {
      std::vector<unsigned> nneigh;
      std::vector<double> cutoff;
      unsigned ncv = getNumberOfArguments();
      if (hill.multivariate) {
        unsigned k = 0;
        Matrix<double> mymatrix(ncv, ncv);
        for (unsigned i = 0; i < ncv; i++) {
          for (unsigned j = i; j < ncv; j++) {
            mymatrix(i, j) = mymatrix(j, i) = hill.sigma[k];
            k++;
          }
        }
        Matrix<double> myinv(ncv, ncv);
        Invert(mymatrix, myinv);
        Matrix<double> myautovec(ncv, ncv);
        std::vector<double> myautoval(ncv);
        diagMat(myinv, myautoval, myautovec);
        double maxautoval = 0.;
        unsigned ind_maxautoval; ind_maxautoval = ncv;
        for (unsigned i = 0; i < ncv; i++) {
          if (myautoval[i] > maxautoval) { maxautoval = myautoval[i]; ind_maxautoval = i; }
        }
        for (unsigned i = 0; i < ncv; i++) {
          cutoff.push_back(std::sqrt(2.0 * dp2cutoff) * std::abs(std::sqrt(maxautoval) * myautovec(i, ind_maxautoval)));
        }
      }
      else {
        for (unsigned i = 0; i < ncv; ++i) {
          cutoff.push_back(std::sqrt(2.0 * dp2cutoff) * hill.sigma[i]);
        }
      }
      for (unsigned i = 0; i < ncv; i++) {
        nneigh.push_back(static_cast<unsigned>(std::ceil(cutoff[i] / BiasGrid_->getDx()[i])));
      }
      return nneigh;
    }
    void DEEPIBP::readcv(IFile& ifilecv) {
      if (!ifilecv.isOpen()) {
        log.printf("[readcv] Error: file is not open!\n");
        return;
      }
      unsigned ncv = getNumberOfArguments();
      std::vector<double> cv(ncv);
      double time;
      std::vector<Value> tmpvalues;
      std::vector<std::vector<double>> allCVData(ncv);
      for (unsigned j = 0; j < ncv; ++j) {
        tmpvalues.emplace_back(this, getPntrToArgument(j)->getName(), false);
      }
      while (scanonecv(&ifilecv, tmpvalues, time, cv)) {
        double timestep = getTimeStep();
        unsigned step = static_cast<unsigned>(time / timestep);
        if (step > initial_stride * num_sub) {
          for (unsigned i = 0; i < ncv; ++i) {
            allCVData[i].push_back(cv[i]);
          }
        }
      }
      for (unsigned i = 0; i < ncv; ++i) {
        if (allCVData[i].empty()) {
          error("CV " + std::to_string(i) + " has no data, cannot compute min/max");
        }
      }
      size_t totalPoints = allCVData[0].size();
      std::string corrected_fes = "fes_wallcorrected_" + std::to_string(n_sub) + ".dat";
      OFile hills;
      hills.link(*this);
      if (auto* grid = dynamic_cast<Grid*>(BiasGrid_.get())) {
        grid->scaleAllValuesAndDerivatives(-1.0);
        hills.open(outhills);
        grid->setMinToZero();
        grid->setOutputFmt(fmt);
        grid->writeToFile(hills);
        hills.close();
        OFile wallOut;
        wallOut.link(*this);
        wallOut.open(corrected_fes);
        IFile wallIn;
        wallIn.link(*this);
        wallIn.open(outhills);
        if (!wallIn.FileExist(outhills))
        {
          error("fes.dat missing");
        }
        wallIn.allowIgnoredFields();
        std::vector<double> wall_fields(ncv), wall_der(ncv);
        double bias;
        while (true) {
          bool ok = true;
          for (unsigned i = 0; i < ncv; ++i) {
            ok &= wallIn.scanField(getPntrToArgument(i)->getName(), wall_fields[i]);
          }
          ok &= wallIn.scanField(getLabel() + ".bias", bias);
          for (unsigned i = 0; i < ncv; ++i) {
            ok &= wallIn.scanField("der_" + getPntrToArgument(i)->getName(), wall_der[i]);
          }
          if (!ok) break;
          wallIn.scanField();
          for (unsigned i = 0; i < ncv; ++i) {
            double cv = wall_fields[i];
            if (cv < lower_wall[i] && lower_wall[i] - cv <= sigma_sub[i]) {
              double d = lower_wall[i] - cv;
              wall_der[i] -= kappa_l[i] * exp_l[i] * std::pow(d, exp_l[i] - 1);
              bias -= kappa_l[i] * std::pow(d, exp_l[i]);
            }
            else if (cv > upper_wall[i] && cv - upper_wall[i] <= sigma_sub[i]) {
              double d = cv - upper_wall[i];
              wall_der[i] -= kappa_u[i] * exp_u[i] * std::pow(d, exp_u[i] - 1);
              bias -= kappa_u[i] * std::pow(d, exp_u[i]);
            }
          }
          for (unsigned i = 0; i < ncv; ++i) {
            wallOut.printField(getPntrToArgument(i)->getName(), wall_fields[i]);
          }
          wallOut.printField("file.free", bias);
          for (unsigned i = 0; i < ncv; ++i) {
            wallOut.printField("der_" + getPntrToArgument(i)->getName(), wall_der[i]);
          }
          wallOut.printField();
        }
      }
      else {
        error("BiasGrid_ is not of type Grid.");
      }
      int numBins = 1 + static_cast<int>(std::log2(totalPoints));
      double kT = 0.595;
      std::vector<std::vector<std::pair<double, double>>> acceptedRanges(ncv);
      if (ncv == 1) {
        for (unsigned i = 0; i < ncv; ++i) {
          if (allCVData[i].empty()) continue;
          double minVal, maxVal, binWidth;
          std::vector<int> histogram = computeHistogram(allCVData[i], numBins, minVal, maxVal, binWidth);
          std::vector<double> probability(numBins);
          for (int b = 0; b < numBins; ++b) {
            probability[b] = static_cast<double>(histogram[b]) / totalPoints;
          }
          std::vector<double> energy(numBins, 0.0);
          double Emin = std::numeric_limits<double>::max();
          for (int b = 0; b < numBins; ++b) {
            double offset = (b + 0.5) * binWidth;
            double binCenter = minVal + offset;
            bool isValidProb = (probability[b] > 0.0);
            bool isWithinBounds = (binCenter >= lower_wall[i] && binCenter <= upper_wall[i]);
            if (isValidProb && isWithinBounds) {
              energy[b] = -kT * std::log(probability[b]);
              if (energy[b] < Emin) {
                Emin = energy[b];
              }
            }
            else if (probability[b] > 0.0) {
              energy[b] = -kT * std::log(probability[b]);
            }
            else {
              energy[b] = std::numeric_limits<double>::infinity();
            }
          }
          for (int b = 0; b < numBins; ++b) {
            if (energy[b] - Emin <= deltaE) {
              double binMin = minVal + b * binWidth;
              double binMax = binMin + binWidth;
              acceptedRanges[i].emplace_back(binMin, binMax);
            }
          }
        }
      }
      else {
        std::vector<double> minVals(ncv), maxVals(ncv), binWidths(ncv);
        for (unsigned i = 0; i < ncv; ++i) {
          auto [minIt, maxIt] = std::minmax_element(allCVData[i].begin(), allCVData[i].end());
          minVals[i] = *minIt;
          maxVals[i] = *maxIt;
          binWidths[i] = (maxVals[i] - minVals[i]) / numBins;
        }
        std::map<std::vector<int>, int> histogram;
        for (size_t p = 0; p < totalPoints; ++p) {
          std::vector<int> binIndex(ncv);
          bool valid = true;
          for (unsigned d = 0; d < ncv; ++d) {
            int idx = static_cast<int>((allCVData[d][p] - minVals[d]) / binWidths[d]);
            if (idx < 0 || idx >= numBins) {
              valid = false;
              break;
            }
            binIndex[d] = idx;
          }
          if (valid) histogram[binIndex]++;
        }
        std::map<std::vector<int>, double> energy;
        double Emin = std::numeric_limits<double>::max();
        for (const auto& [bins, count] : histogram) {
          double prob = static_cast<double>(count) / totalPoints;
          double e = (prob > 0.0) ? -kT * std::log(prob) : std::numeric_limits<double>::infinity();
          energy[bins] = e;
          if (prob > 0.0 && e < Emin) Emin = e;
        }
        for (const auto& [bins, e] : energy) {
          if (e - Emin <= deltaE) {
            for (unsigned d = 0; d < ncv; ++d) {
              double binMin = minVals[d] + bins[d] * binWidths[d];
              double binMax = binMin + binWidths[d];
              acceptedRanges[d].emplace_back(binMin, binMax);
            }
          }
        }
      }
      IFile fesIn;
      fesIn.link(*this);
      if (!fesIn.FileExist(corrected_fes)) {
        error("FES file " + outhills + " not found.");
      }
      fesIn.open(corrected_fes);
      fesIn.allowIgnoredFields();
      OFile fesOut;
      fesOut.link(*this);
      fesOut.open(outfiled_fes);
      std::vector<double> fields(ncv);
      while (true) {
        bool ok = true;
        for (unsigned i = 0; i < ncv; ++i) {
          std::string name = getPntrToArgument(i)->getName();
          if (!fesIn.scanField(name, fields[i])) {
            ok = false;
            break;
          }
        }
        double bias;
        ok &= fesIn.scanField("file.free", bias);
        std::vector<double> derivatives(ncv);
        for (unsigned i = 0; i < ncv; ++i) {
          ok &= fesIn.scanField("der_" + getPntrToArgument(i)->getName(), derivatives[i]);
        }
        fesIn.scanField();
        if (!ok) break;
        bool inAcceptedRegion = true;
        for (unsigned i = 0; i < ncv; ++i) {
          double val = fields[i];
          if (val < (lower_wall[i] - 0.5 * sigma_sub[i]) || val >(upper_wall[i] + 0.5 * sigma_sub[i])) {
            inAcceptedRegion = false;
            break;
          }
          bool matched = false;
          for (const auto& range : acceptedRanges[i]) {
            if (val >= range.first && val < range.second) {
              matched = true;
              break;
            }
          }
          if (!matched) {
            inAcceptedRegion = false;
            break;
          }
        }
        if (inAcceptedRegion) {
          for (unsigned i = 0; i < ncv; ++i) {
            fesOut.printField(getPntrToArgument(i)->getName(), fields[i]);
          }
          fesOut.printField("file.free", bias);
          for (unsigned i = 0; i < ncv; ++i) {
            fesOut.printField("der_" + getPntrToArgument(i)->getName(), derivatives[i]);
          }
          fesOut.printField();
        }
      }
    }
    void DEEPIBP::addGaussian(const Gaussian& hill)
    {
      const unsigned dbg_rank = comm.Get_rank();
      const unsigned dbg_size = comm.Get_size();
      if (grid_) {
        size_t ncv = getNumberOfArguments();
        std::vector<unsigned> nneighb = getGaussianSupport(hill);
        std::vector<Grid::index_t> neighbors = BiasGrid_->getNeighbors(hill.center, nneighb);
        std::vector<double> der(ncv);
        std::vector<double> xx(ncv);
        if (comm.Get_size() == 1) {
          std::vector<double> dp(ncv);
          for (size_t i = 0; i < neighbors.size(); ++i) {
            Grid::index_t ineigh = neighbors[i];
            for (unsigned j = 0; j < ncv; ++j) der[j] = 0.0;
            BiasGrid_->getPoint(ineigh, xx);
            double bias = evaluateGaussianAndDerivatives(xx, hill, der, dp);
            BiasGrid_->addValueAndDerivatives(ineigh, bias, der);
            log.printf("point %d ,hills %f", ineigh, bias);
          }
        }
        else {
          unsigned stride = comm.Get_size();
          unsigned rank = comm.Get_rank();
          std::vector<double> allder(ncv * neighbors.size(), 0.0);
          std::vector<double> n_der(ncv, 0.0);
          std::vector<double> allbias(neighbors.size(), 0.0);
          std::vector<double> dp(ncv);
          for (unsigned i = rank; i < neighbors.size(); i += stride) {
            Grid::index_t ineigh = neighbors[i];
            for (unsigned j = 0; j < ncv; ++j) n_der[j] = 0.0;
            BiasGrid_->getPoint(ineigh, xx);
            allbias[i] = evaluateGaussianAndDerivatives(xx, hill, n_der, dp);
            for (unsigned j = 0; j < ncv; j++) allder[ncv * i + j] = n_der[j];
          }
          comm.Sum(allbias);
          comm.Sum(allder);
          for (unsigned i = 0; i < neighbors.size(); ++i) {
            Grid::index_t ineigh = neighbors[i];
            for (unsigned j = 0; j < ncv; ++j) der[j] = allder[ncv * i + j];
            BiasGrid_->addValueAndDerivatives(ineigh, allbias[i], der);
          }
        }
        if (guess) {
          ++hills_count_;
          if (stride_hap > 0) {
            if (hills_count_ % (unsigned long)stride_hap == 0UL) {
              const std::string fname = "hap_" + std::to_string(snapshot_idx_) + ".dat";
              writeFESSnapshot_(fname, true);
              ++snapshot_idx_;
            }
          }
        }
      }
      else
        hills_.push_back(hill);
    }
    bool DEEPIBP::scanOneHill(IFile* ifile, std::vector<Value>& tmpvalues, std::vector<double>& center, std::vector<double>& sigma, double& height, bool& multivariate)
    {
      double dummy;
      multivariate = false;
      if (ifile->scanField("time", dummy)) {
        unsigned ncv = tmpvalues.size();
        for (unsigned i = 0; i < ncv; ++i) {
          ifile->scanField(&tmpvalues[i]);
          if (tmpvalues[i].isPeriodic() && !getPntrToArgument(i)->isPeriodic()) {
            error("in hills file periodicity for variable " + tmpvalues[i].getName() + " does not match periodicity in input");
          }
          else if (tmpvalues[i].isPeriodic()) {
            std::string imin, imax; tmpvalues[i].getDomain(imin, imax);
            std::string rmin, rmax; getPntrToArgument(i)->getDomain(rmin, rmax);
            if (imin != rmin || imax != rmax) {
              error("in hills file periodicity for variable " + tmpvalues[i].getName() + " does not match periodicity in input");
            }
          }
          center[i] = tmpvalues[i].get();
        }
        std::string ktype = "stretched-gaussian";
        if (ifile->FieldExist("kerneltype")) ifile->scanField("kerneltype", ktype);
        else if (ktype != "stretched-gaussian") {
          error("non Gaussian kernels are not supported in MetaD");
        }
        std::string sss;
        ifile->scanField("multivariate", sss);
        if (sss == "true") multivariate = true;
        else if (sss == "false") multivariate = false;
        else plumed_merror("cannot parse multivariate = " + sss);
        if (multivariate) {
          sigma.resize(ncv * (ncv + 1) / 2);
          Matrix<double> upper(ncv, ncv);
          Matrix<double> lower(ncv, ncv);
          for (unsigned i = 0; i < ncv; i++) {
            for (unsigned j = 0; j < ncv - i; j++) {
              ifile->scanField("sigma_" + getPntrToArgument(j + i)->getName() + "_" + getPntrToArgument(j)->getName(), lower(j + i, j));
              upper(j, j + i) = lower(j + i, j);
            }
          }
          Matrix<double> mymult(ncv, ncv);
          Matrix<double> invmatrix(ncv, ncv);
          mult(lower, upper, mymult);
          Invert(mymult, invmatrix);
          unsigned k = 0;
          for (unsigned i = 0; i < ncv; i++) {
            for (unsigned j = i; j < ncv; j++) {
              sigma[k] = invmatrix(i, j);
              k++;
            }
          }
        }
        else {
          for (unsigned i = 0; i < ncv; ++i) {
            ifile->scanField("sigma_" + getPntrToArgument(i)->getName(), sigma[i]);
          }
        }
        ifile->scanField("height", height);
        ifile->scanField("biasf", dummy);
        if (ifile->FieldExist("clock")) ifile->scanField("clock", dummy);
        if (ifile->FieldExist("lower_int")) ifile->scanField("lower_int", dummy);
        if (ifile->FieldExist("upper_int")) ifile->scanField("upper_int", dummy);
        ifile->scanField();
        return true;
      }
      else {
        return false;
      }
    }
    std::vector<int> DEEPIBP::computeHistogram(const std::vector<double>& data, int numBins, double& minVal, double& maxVal, double& binWidth) {
      std::vector<int> resultHistogram;
      if (data.empty()) {
        error("computeHistogram error: Input data is empty.");
      }
      if (numBins <= 0) {
        error("computeHistogram error: numBins must be positive.");
      }
      double currentMin = data[0];
      double currentMax = data[0];
      for (const double& val : data) {
        if (val < currentMin) currentMin = val;
        if (val > currentMax) currentMax = val;
      }
      minVal = currentMin;
      maxVal = currentMax;
      if (std::abs(currentMax - currentMin) < 1e-12) {
        log.printf("All data points are equal. Using single bin.");
        binWidth = 1.0;
        resultHistogram.resize(1, static_cast<int>(data.size()));
        return resultHistogram;
      }
      binWidth = (currentMax - currentMin) / static_cast<double>(numBins);
      log.printf("Computed bin width: %.6f\n", binWidth);
      resultHistogram.assign(numBins, 0);
      int outOfRangeCount = 0;
      for (size_t i = 0; i < data.size(); ++i) {
        double val = data[i];
        int binIdx = static_cast<int>((val - currentMin) / binWidth);
        if (binIdx >= numBins) {
          binIdx = numBins - 1;
          ++outOfRangeCount;
        }
        resultHistogram[binIdx]++;
      }
      log.printf(" Histogram bins:\n");
      for (int i = 0; i < numBins; ++i) {
        double binStart = currentMin + i * binWidth;
        double binEnd = binStart + binWidth;
        log.printf("  Bin %2d [%.2f, %.2f): %d\n", i, binStart, binEnd, resultHistogram[i]);
      }
      if (outOfRangeCount > 0) {
        log.printf("[Warning] %d data points were clipped into the last bin.\n", outOfRangeCount);
      }
      return resultHistogram;
    }
    void DEEPIBP::writeGaussian(const Gaussian& hill, OFile& file)
    {
      unsigned ncv = getNumberOfArguments();
      file.printField("time", getTimeStep() * getStep());
      for (unsigned i = 0; i < ncv; ++i) {
        file.printField(getPntrToArgument(i), hill.center[i]);
      }
      file.printField("kerneltype", "stretched-gaussian");
      if (hill.multivariate) {
        file.printField("multivariate", "true");
        Matrix<double> mymatrix(ncv, ncv);
        unsigned k = 0;
        for (unsigned i = 0; i < ncv; i++) {
          for (unsigned j = i; j < ncv; j++) {
            mymatrix(i, j) = mymatrix(j, i) = hill.sigma[k];
            k++;
          }
        }
        Matrix<double> invmatrix(ncv, ncv);
        Invert(mymatrix, invmatrix);
        for (unsigned i = 0; i < ncv; i++) {
          for (unsigned j = i; j < ncv; j++) {
            invmatrix(i, j) = invmatrix(j, i);
          }
        }
        Matrix<double> lower(ncv, ncv);
        cholesky(invmatrix, lower);
        for (unsigned i = 0; i < ncv; i++) {
          for (unsigned j = 0; j < ncv - i; j++) {
            file.printField("sigma_" + getPntrToArgument(j + i)->getName() + "_" + getPntrToArgument(j)->getName(), lower(j + i, j));
          }
        }
      }
      else {
        file.printField("multivariate", "false");
        for (unsigned i = 0; i < ncv; ++i)
          file.printField("sigma_" + getPntrToArgument(i)->getName(), hill.sigma[i]);
      }
      double height = hill.height;
      if (welltemp_ && biasf_ > 1.0) height *= biasf_ / (biasf_ - 1.0);
      file.printField("height", height).printField("biasf", biasf_);
      file.printField();
    }
    double DEEPIBP::getBiasAndDerivatives(const std::vector<double>& cv, std::vector<double>& der)
    {
      unsigned ncv = getNumberOfArguments();
      double bias = 0.0;
      if (grid_) {
        std::vector<double> vder(ncv);
        bias = BiasGrid_->getValueAndDerivatives(cv, vder);
        for (unsigned i = 0; i < ncv; i++) der[i] = vder[i];
      }
      else {
        unsigned nt = OpenMP::getNumThreads();
        unsigned stride = comm.Get_size();
        unsigned rank = comm.Get_rank();
        if (hills_.size() < 2 * nt * stride || nt == 1) {
          std::vector<double> dp(ncv);
          for (unsigned i = rank; i < hills_.size(); i += stride) {
            bias += evaluateGaussianAndDerivatives(cv, hills_[i], der, dp);
          }
        }
        else {
#pragma omp parallel num_threads(nt)
          {
            std::vector<double> omp_deriv(ncv, 0.);
            std::vector<double> dp(ncv);
#pragma omp for reduction(+:bias) nowait
            for (unsigned i = rank; i < hills_.size(); i += stride) {
              bias += evaluateGaussianAndDerivatives(cv, hills_[i], omp_deriv, dp);
            }
#pragma omp critical
            for (unsigned i = 0; i < ncv; i++) der[i] += omp_deriv[i];
          }
        }
        comm.Sum(bias);
        comm.Sum(der);
      }
      return bias;
    }
    void DEEPIBP::writeFESSnapshot_(const std::string& fname, bool restore_after) {
      auto* grid = dynamic_cast<Grid*>(BiasGrid_.get());
      if (!grid) error("BiasGrid_ is not of type Grid.");
      grid->scaleAllValuesAndDerivatives(-1.0);
      if (minTOzero) grid->setMinToZero();
        OFile fes;
        fes.link(*this).open(fname);
        BiasGrid_->writeToFile(fes); fes.close();
      if (restore_after) grid->scaleAllValuesAndDerivatives(-1.0);
    }
    bool DEEPIBP::scanonecv(IFile* ifile, std::vector<Value>& tmpvalues, double& time, std::vector<double>& cv) {
      if (ifile->scanField("time", time)) {
        double timestep = getTimeStep();
        unsigned step = static_cast<unsigned>(time / timestep);
        const unsigned stride_threshold = initial_stride * num_sub;
        unsigned ncv = tmpvalues.size();
        if (cv.size() != ncv) {
          log.printf(" Warning: cv size (%lu) does not match tmpvalues size (%u). Resizing...\n",
            cv.size(), ncv);
          cv.resize(ncv);
        }
        for (unsigned i = 0; i < ncv; ++i) {
          ifile->scanField(&tmpvalues[i]);
          if (tmpvalues.empty()) {
            error("[scanonecv] Error: tmpvalues vector is empty, cannot scan CVs.");
            return false;
          }
        }
        ifile->scanField();
        if (step > stride_threshold) {
          for (unsigned i = 0; i < ncv; ++i) {
            cv[i] = tmpvalues[i].get();
          }
        }
        return true;
      }
      else {
        return false;
      }
    }
    void DEEPIBP::readGaussians(IFile* ifile)
    {
      unsigned ncv = getNumberOfArguments();
      std::vector<double> center(ncv);
      std::vector<double> sigma(ncv);
      double height;
      int nhills = 0;
      bool multivariate = false;
      std::vector<Value> tmpvalues;
      for (unsigned j = 0; j < getNumberOfArguments(); ++j) tmpvalues.push_back(Value(this, getPntrToArgument(j)->getName(), false));
      while (scanOneHill(ifile, tmpvalues, center, sigma, height, multivariate))
      {
        nhills++;
        if (welltemp_ && biasf_ > 1.0) height *= (biasf_ - 1.0) / biasf_;
        addGaussian(Gaussian(multivariate, height, center, sigma));
      }
      populated_gaussian = static_cast<unsigned>(nhills);
      log.printf("      %d Gaussians read\n", nhills);
    }
    int DEEPIBP::countHitsForDim(int dim, double anchor, int op, double D) const {
      if (dim < 0 || dim >= static_cast<int>(monitor_cvs_.size())) return 0;
      int arg_index = monitor_cvs_[dim];
      if (hill_cv_at_add_.empty()) return 0;
      double thr = (op >= 0) ? (anchor + D) : (anchor - D);
      int hits = 0;
      for (const auto& cv : hill_cv_at_add_) {
        double v = cv[arg_index];
        if (op >= 0) {
          if (v >= thr) ++hits;
        }
        else {
          if (v <= thr) ++hits;
        }
      }
      return hits;
    }
    void DEEPIBP::buildBoxFromCustomAnchorAndD(double D0, double D1) {
      if (monitor_cvs_.size() < 2 || custom_thr_.size() < 2) {
        buildBoxFromCustomThr();
        return;
      }
      if (a_box_.size() < 2) a_box_.assign(2, 0.0);
      if (b_box_.size() < 2) b_box_.assign(2, 0.0);
      double anchor0 = custom_thr_[0];
      double anchor1 = custom_thr_[1];
      int op0 = (custom_op_.size() > 0 ? custom_op_[0] : +1);
      int op1 = (custom_op_.size() > 1 ? custom_op_[1] : +1);
      double a0, b0, a1, b1;
      custom_range_from_anchor(anchor0, op0, (op0 >= 0 ? D0 : -D0), a0, b0);
      custom_range_from_anchor(anchor1, op1, (op1 >= 0 ? D1 : -D1), a1, b1);
      a_box_[0] = a0; b_box_[0] = b0;
      a_box_[1] = a1; b_box_[1] = b1;
    }
    void DEEPIBP::adjustKAndDWhenKNotReached2D(double& D0, double& D1, int& K_used) {
      const std::size_t nmon = monitor_cvs_.size();
      if (!kl_region || nmon < 2) return;
      if (hill_cv_at_add_.empty()) return;
      int K0 = (use_dynamic_K_ && dynamic_K_ > 0)
        ? static_cast<int>(std::ceil(dynamic_K_))
        : hit_trigger_times_;
      if (K0 < min_K_) K0 = min_K_;
      double D_orig[2] = {
        std::fabs(dcv_[0]),
        std::fabs(dcv_[1])
      };
      double sigma0 = std::fabs(sigma_sub[monitor_cvs_[0]]);
      double sigma1 = std::fabs(sigma_sub[monitor_cvs_[1]]);
      if (sigma0 <= 0.0) sigma0 = 1e-6;
      if (sigma1 <= 0.0) sigma1 = 1e-6;
      double D_best[2] = { D_orig[0], D_orig[1] };
      int K_best = K0;
      bool found = false;
      double anchors[2] = {
        custom_thr_.empty() ? 0.0 : custom_thr_[0],
        custom_thr_.size() > 1 ? custom_thr_[1] : 0.0
      };
      int ops[2] = {
        (custom_op_.size() > 0 ? custom_op_[0] : +1),
        (custom_op_.size() > 1 ? custom_op_[1] : +1)
      };
      const int N_D_STEPS = 50;
      for (int K_try = K0; K_try >= min_K_; K_try -= 2) {
        double D_cand_dim[2] = { D_orig[0], D_orig[1] };
        bool thisK_ok = true;
        for (int d = 0; d < 2; ++d) {
          double Dmax = D_orig[d];
          double sig = (d == 0 ? sigma0 : sigma1);
          bool found_dim = false;
          if (Dmax <= sig) {
            int hits_at_Dmax = countHitsForDim(d, anchors[d], ops[d], Dmax);
            if (hits_at_Dmax >= K_try) {
              D_cand_dim[d] = Dmax;
              found_dim = true;
            }
          }
          else {
            for (int s = 0; s <= N_D_STEPS; ++s) {
              double alpha = static_cast<double>(s) / N_D_STEPS;
              double Dcand = Dmax - alpha * (Dmax - sig);
              int hits = countHitsForDim(d, anchors[d], ops[d], Dcand);
              if (hits >= K_try) {
                D_cand_dim[d] = Dcand;
                found_dim = true;
                break;
              }
            }
          }
          if (!found_dim) {
            thisK_ok = false;
            break;
          }
        }
        if (thisK_ok) {
          D_best[0] = D_cand_dim[0];
          D_best[1] = D_cand_dim[1];
          K_best = K_try;
          found = true;
          break;
        }
      }
      if (!found) {
        for (int d = 0; d < 2; ++d) {
          double sig = (d == 0 ? sigma0 : sigma1);
          double Dmax = D_orig[d];
          double Dtmp = std::max(sig, std::min(Dmax, delta_raw_[d]));
          int hits = countHitsForDim(d, anchors[d], ops[d], Dtmp);
          D_best[d] = Dtmp;
          if (hits > K_best) K_best = hits;
        }
        if (K_best < min_K_) K_best = min_K_;
      }
      D0 = D_best[0];
      D1 = D_best[1];
      K_used = K_best;
    }
    void DEEPIBP::processCV(IFile& ifilecv) {
      log.printf(" Start processing CV\n");
      unsigned ncv = getNumberOfArguments();
      std::vector<double> cv(ncv);
      double time;
      std::vector<Value> tmpvalues;
      std::vector<std::vector<double>> allCVData(ncv);
      double Kb = plumed.getAtoms().getKBoltzmann();
      double KT = plumed.getAtoms().getKBoltzmann() * temp_;
      for (unsigned j = 0; j < ncv; ++j) {
        tmpvalues.emplace_back(this, getPntrToArgument(j)->getName(), false);
      }
      while (scanonecv(&ifilecv, tmpvalues, time, cv)) {
        double timestep = getTimeStep();
        unsigned step = static_cast<unsigned>(time / timestep);
        if (step > initial_stride * num_sub) {
          for (unsigned i = 0; i < ncv; ++i) {
            allCVData[i].push_back(cv[i]);
          }
        }
      }
      size_t totalPoints = allCVData[0].size();
      std::vector<double> minVals(ncv), maxVals(ncv);
      for (unsigned i = 0; i < ncv; ++i) {
        if (allCVData[i].empty()) {
          error("CV " + std::to_string(i) + " has no data, cannot compute min/max");
          minVals[i] = 0.0;
          maxVals[i] = 0.0;
          continue;
        }
        auto [minIt, maxIt] = std::minmax_element(allCVData[i].begin(), allCVData[i].end());
        minVals[i] = *minIt;
        maxVals[i] = *maxIt;
      }
      OFile hills;
      hills.link(*this);
      std::string corrected_fes = "fes_wallcorrected_" + std::to_string(n_sub) + ".dat";
      if (auto* grid = dynamic_cast<Grid*>(BiasGrid_.get())) {
        grid->scaleAllValuesAndDerivatives(-1.0);
        hills.open(outhills);
        grid->setMinToZero();
        grid->setOutputFmt(fmt);
        grid->writeToFile(hills);
        hills.close();
        OFile wallOut; wallOut.link(*this); wallOut.open(corrected_fes);
        IFile wallIn; wallIn.link(*this); wallIn.open(outhills);
        if (!wallIn.FileExist(outhills)) { error("fes.dat missing"); }
        wallIn.allowIgnoredFields();
        std::vector<double> wall_fields(ncv), wall_der(ncv);
        double bias = 0.0;
        while (true) {
          bool ok = true;
          for (unsigned i = 0; i < ncv; ++i) ok &= wallIn.scanField(getPntrToArgument(i)->getName(), wall_fields[i]);
          ok &= wallIn.scanField(getLabel() + ".bias", bias);
          for (unsigned i = 0; i < ncv; ++i) ok &= wallIn.scanField("der_" + getPntrToArgument(i)->getName(), wall_der[i]);
          if (!ok) break;
          wallIn.scanField();
          for (unsigned i = 0; i < ncv; ++i) {
            double cvv = wall_fields[i];
            if (cvv < lower_wall[i] && lower_wall[i] - cvv <= sigma_sub[i]) {
              double d = lower_wall[i] - cvv;
              wall_der[i] -= kappa_l[i] * exp_l[i] * std::pow(d, exp_l[i] - 1);
              bias -= kappa_l[i] * std::pow(d, exp_l[i]);
            }
            else if (cvv > upper_wall[i] && cvv - upper_wall[i] <= sigma_sub[i]) {
              double d = cvv - upper_wall[i];
              wall_der[i] -= kappa_u[i] * exp_u[i] * std::pow(d, exp_u[i] - 1);
              bias -= kappa_u[i] * std::pow(d, exp_u[i]);
            }
          }
          for (unsigned i = 0; i < ncv; ++i) wallOut.printField(getPntrToArgument(i)->getName(), wall_fields[i]);
          wallOut.printField("file.free", bias);
          for (unsigned i = 0; i < ncv; ++i) wallOut.printField("der_" + getPntrToArgument(i)->getName(), wall_der[i]);
          wallOut.printField();
        }
      }
      else {
        error("BiasGrid_ is not of type Grid.");
      }
      if (bin_hap.size() < ncv) {
        error("bin_hap size < ncv. Please provide bin counts for each CV.");
      }
      const int N_grid_x = bin_hap[0];
      const int N_grid_y = (ncv == 2 ? bin_hap[1] : 1);
      if (N_grid_x < 2 || (ncv == 2 && N_grid_y < 2)) {
        error("bin_hap entries must be >= 2 for each dimension.");
      }
      std::vector<double> grid_x, grid_y;
      const double phi_min = gmin[0];
      const double phi_max = gmax[0];
      double psi_min = 0.0, psi_max = 0.0;
      if (ncv == 2) { psi_min = gmin[1]; psi_max = gmax[1]; }
      const double dx = (phi_max - phi_min) / (N_grid_x - 1);
      const double dy = (ncv == 2 ? (psi_max - psi_min) / (N_grid_y - 1) : 1.0);
      grid_x.resize(N_grid_x);
      for (int i = 0; i < N_grid_x; ++i) grid_x[i] = phi_min + i * dx;
      if (ncv == 2) {
        grid_y.resize(N_grid_y);
        for (int j = 0; j < N_grid_y; ++j) grid_y[j] = psi_min + j * dy;
      }
      std::vector<double> prob((ncv == 1) ? N_grid_x : N_grid_x * N_grid_y, 0.0);
      const size_t nsamp = allCVData[0].size();
      if (nsamp == 0) error("No CV samples for KDE.");
      auto variance_ddof1 = [](const std::vector<double>& v) {
        const size_t n = v.size();
        if (n <= 1) return 0.0;
        double m = 0.0; for (double x : v) m += x; m /= double(n);
        double s = 0.0; for (double x : v) { double d = x - m; s += d * d; }
        return s / double(n - 1);
        };
      if (ncv == 1) {
        const double var = std::max(variance_ddof1(allCVData[0]), 1e-16);
        const double stdv = std::sqrt(var);
        const double factor = std::pow(double(nsamp), -1.0 / 5.0);
        const double h = std::max(factor * stdv, 1e-8);
        const double norm = 1.0 / (double(nsamp) * h * std::sqrt(2.0 * kPi));
        for (int i = 0; i < N_grid_x; ++i) {
          double x = grid_x[i];
          double sum = 0.0;
          for (size_t k = 0; k < nsamp; ++k) {
            double u = (x - allCVData[0][k]) / h;
            sum += std::exp(-0.5 * u * u);
          }
          prob[i] = norm * sum;
        }
      }
      else if (ncv == 2) {
        double mx = 0.0, my = 0.0;
        for (size_t k = 0; k < nsamp; ++k) { mx += allCVData[0][k]; my += allCVData[1][k]; }
        mx /= double(nsamp); my /= double(nsamp);
        double cxx = 0.0, cyy = 0.0, cxy = 0.0;
        for (size_t k = 0; k < nsamp; ++k) {
          double dxs = allCVData[0][k] - mx;
          double dys = allCVData[1][k] - my;
          cxx += dxs * dxs; cyy += dys * dys; cxy += dxs * dys;
        }
        const double denom = std::max<double>(nsamp - 1, 1);
        cxx /= denom; cyy /= denom; cxy /= denom;
        const double factor = std::pow(double(nsamp), -1.0 / 6.0);
        const double a = factor * factor * cxx;
        const double b = factor * factor * cxy;
        const double c = factor * factor * cyy;
        const double detH = std::max(a * c - b * b, 1e-24);
        const double inv00 = c / detH;
        const double inv01 = -b / detH;
        const double inv10 = -b / detH;
        const double inv11 = a / detH;
        const double norm = 1.0 / (double(nsamp) * 2.0 * kPi * std::sqrt(detH));
        for (int i = 0; i < N_grid_x; ++i) {
          for (int j = 0; j < N_grid_y; ++j) {
            const double x = grid_x[i], y = grid_y[j];
            double sum = 0.0;
            for (size_t k = 0; k < nsamp; ++k) {
              const double dxs = x - allCVData[0][k];
              const double dys = y - allCVData[1][k];
              const double q0 = inv00 * dxs + inv01 * dys;
              const double q1 = inv10 * dxs + inv11 * dys;
              const double Q = dxs * q0 + dys * q1;
              sum += std::exp(-0.5 * Q);
            }
            prob[i * N_grid_y + j] = norm * sum;
          }
        }
      }
      else {
        error("KDE only implemented for ncv=1 or ncv=2.");
      }
      double integral = 0.0;
      if (ncv == 1) {
        for (int i = 0; i < N_grid_x; ++i) integral += prob[i] * dx;
        integral = std::max(integral, 1e-24);
        for (int i = 0; i < N_grid_x; ++i) prob[i] /= integral;
      }
      else {
        for (int i = 0; i < N_grid_x; ++i)
          for (int j = 0; j < N_grid_y; ++j)
            integral += prob[i * N_grid_y + j] * dx * dy;
        integral = std::max(integral, 1e-24);
        for (int i = 0; i < N_grid_x; ++i)
          for (int j = 0; j < N_grid_y; ++j)
            prob[i * N_grid_y + j] /= integral;
      }
      log.printf("KDE_normalization = %g\n", integral);
      {
        OFile kdeOut; kdeOut.link(*this); kdeOut.open("kde_density_data.dat");
        if (ncv == 1) {
          for (int i = 0; i < N_grid_x; ++i) {
            kdeOut.printField(getPntrToArgument(0)->getName(), grid_x[i])
              .printField("density", prob[i]).printField();
          }
        }
        else {
          for (int i = 0; i < N_grid_x; ++i) {
            for (int j = 0; j < N_grid_y; ++j) {
              kdeOut.printField(getPntrToArgument(0)->getName(), grid_x[i])
                .printField(getPntrToArgument(1)->getName(), grid_y[j])
                .printField("density", prob[i * N_grid_y + j]).printField();
            }
          }
        }
      }
      std::vector<double> logp(prob.size());
      for (size_t t = 0; t < prob.size(); ++t) logp[t] = std::log(prob[t] + 1e-300);
      std::vector<double> dlogp_dphi(prob.size(), 0.0), dlogp_dpsi(prob.size(), 0.0);
      if (ncv == 1) {
        for (int i = 0; i < N_grid_x; i++) {
          double dph;
          if (i == 0) dph = (logp[i + 1] - logp[i]) / dx;
          else if (i == N_grid_x - 1) dph = (logp[i] - logp[i - 1]) / dx;
          else dph = (logp[i + 1] - logp[i - 1]) / (2.0 * dx);
          dlogp_dphi[i] = dph;
        }
      }
      else {
        for (int i = 0; i < N_grid_x; i++) {
          for (int j = 0; j < N_grid_y; j++) {
            const int idx = i * N_grid_y + j;
            double dph, dps;
            if (i == 0) dph = (logp[idx + N_grid_y] - logp[idx]) / dx;
            else if (i == N_grid_x - 1) dph = (logp[idx] - logp[idx - N_grid_y]) / dx;
            else dph = (logp[idx + N_grid_y] - logp[idx - N_grid_y]) / (2.0 * dx);
            if (j == 0) dps = (logp[idx + 1] - logp[idx]) / dy;
            else if (j == N_grid_y - 1) dps = (logp[idx] - logp[idx - 1]) / dy;
            else dps = (logp[idx + 1] - logp[idx - 1]) / (2.0 * dy);
            dlogp_dphi[idx] = dph;
            dlogp_dpsi[idx] = dps;
          }
        }
      }
      double beta = 1.0 / KT;
      std::vector<double> force_phi(prob.size(), 0.0), force_psi(prob.size(), 0.0);
      for (size_t t = 0; t < prob.size(); ++t) {
        force_phi[t] = -(1.0 / beta) * dlogp_dphi[t];
        if (ncv == 2) force_psi[t] = -(1.0 / beta) * dlogp_dpsi[t];
      }
      {
        OFile fOut;
        fOut.link(*this);
        fOut.open("first_term_forces.dat");
        if (ncv == 1) {
          for (int i = 0; i < N_grid_x; ++i) {
            const int idx = i;
            fOut.printField(getPntrToArgument(0)->getName(), grid_x[i])
              .printField("density", prob[idx])
              .printField("force_" + getPntrToArgument(0)->getName(), force_phi[idx])
              .printField();
          }
        }
        else {
          for (int i = 0; i < N_grid_x; ++i) {
            for (int j = 0; j < N_grid_y; ++j) {
              const int idx = i * N_grid_y + j;
              fOut.printField(getPntrToArgument(0)->getName(), grid_x[i])
                .printField(getPntrToArgument(1)->getName(), grid_y[j])
                .printField("density", prob[idx])
                .printField("force_" + getPntrToArgument(0)->getName(), force_phi[idx])
                .printField("force_" + getPntrToArgument(1)->getName(), force_psi[idx])
                .printField();
            }
          }
        }
      }
      log.printf(" first_term_forces.dat has been written \n");
      std::vector<double> fes_bias_grid((ncv == 1) ? N_grid_x : N_grid_x * N_grid_y, 0.0);
      std::vector<double> fes_der_phi_grid((ncv == 1) ? N_grid_x : N_grid_x * N_grid_y, 0.0);
      std::vector<double> fes_der_psi_grid((ncv == 1) ? N_grid_x : N_grid_x * N_grid_y, 0.0);
      {
        IFile fesIn; fesIn.link(*this); fesIn.open(corrected_fes);
        fesIn.allowIgnoredFields();
        std::vector<double> fields(ncv), ders(ncv);
        double bias_val = 0.0;
        while (true) {
          bool ok = true;
          for (unsigned i = 0; i < ncv; ++i)
            ok &= fesIn.scanField(getPntrToArgument(i)->getName(), fields[i]);
          ok &= fesIn.scanField("file.free", bias_val);
          for (unsigned i = 0; i < ncv; ++i)
            ok &= fesIn.scanField("der_" + getPntrToArgument(i)->getName(), ders[i]);
          if (!ok) break;
          fesIn.scanField();
          if (ncv == 1) {
            int i_idx = (int)std::llround((fields[0] - phi_min) / dx);
            if (i_idx >= 0 && i_idx < N_grid_x) {
              fes_bias_grid[i_idx] = bias_val;
              fes_der_phi_grid[i_idx] = ders[0];
            }
          }
          else {
            int i_idx = (int)std::llround((fields[0] - phi_min) / dx);
            int j_idx = (int)std::llround((fields[1] - psi_min) / dy);
            if (i_idx >= 0 && i_idx < N_grid_x && j_idx >= 0 && j_idx < N_grid_y) {
              const int idx = i_idx * N_grid_y + j_idx;
              fes_bias_grid[idx] = bias_val;
              fes_der_phi_grid[idx] = ders[0];
              fes_der_psi_grid[idx] = ders[1];
            }
          }
        }
      }
      const double max_density = *std::max_element(prob.begin(), prob.end());
      const double threshold = 0.368 * max_density;
      OFile out_boundary, out_density;
      out_boundary.link(*this); out_boundary.open("fes_corrected_boundary.dat");
      out_density.link(*this); out_density.open("fes_corrected_density_boundary.dat");
      std::vector<double> wall_der(ncv, 0.0);
      const std::string name_phi = getPntrToArgument(0)->getName();
      const std::string name_psi = (ncv == 2 ? getPntrToArgument(1)->getName() : std::string());
      if (ncv == 1) {
        const double phi_lo_keep = lower_wall[0] + 0.1;
        const double phi_hi_keep = upper_wall[0] - 0.1;
        for (int i = 0; i < N_grid_x; i++) {
          const int idx = i;
          const double phi = grid_x[i];
          const double bias_val = fes_bias_grid[idx];
          if (phi < phi_lo_keep || phi > phi_hi_keep) continue;
          const double corrected_phi = wall_der[0] + force_phi[idx] + fes_der_phi_grid[idx];
          out_boundary.printField(name_phi, phi)
            .printField("bias", bias_val)
            .printField(std::string("der_") + name_phi, corrected_phi)
            .printField();
          if (prob[idx] >= threshold) {
            out_density.printField(name_phi, phi)
              .printField("bias", bias_val)
              .printField(std::string("der_") + name_phi, corrected_phi)
              .printField();
          }
        }
      }
      else {
        const double phi_lo_keep = lower_wall[0] + 0.1;
        const double phi_hi_keep = upper_wall[0] - 0.1;
        const double psi_lo_keep = lower_wall[1] + 0.1;
        const double psi_hi_keep = upper_wall[1] - 0.1;
        for (int i = 0; i < N_grid_x; i++) {
          for (int j = 0; j < N_grid_y; j++) {
            const int idx = i * N_grid_y + j;
            const double phi = grid_x[i], psi = grid_y[j];
            const double bias_val = fes_bias_grid[idx];
            if (phi < phi_lo_keep || phi > phi_hi_keep ||
              psi < psi_lo_keep || psi > psi_hi_keep) continue;
            const double corrected_phi = wall_der[0] + force_phi[idx] + fes_der_phi_grid[idx];
            const double corrected_psi = wall_der[1] + force_psi[idx] + fes_der_psi_grid[idx];
            out_boundary.printField(name_phi, phi)
              .printField(name_psi, psi)
              .printField("bias", bias_val)
              .printField(std::string("der_") + name_phi, corrected_phi)
              .printField(std::string("der_") + name_psi, corrected_psi)
              .printField();
            if (prob[idx] >= threshold) {
              out_density.printField(name_phi, phi)
                .printField(name_psi, psi)
                .printField("bias", bias_val)
                .printField(std::string("der_") + name_phi, corrected_phi)
                .printField(std::string("der_") + name_psi, corrected_psi)
                .printField();
            }
          }
        }
      }
      log.printf("fes_corrected_boundary.dat & fes_corrected_density_boundary.dat 已生成\n");
    }
    double DEEPIBP::gaussian_kde(const std::vector<double>& data, double x, double h) {
      if (data.empty() || !std::isfinite(h) || h <= 0.0) {
        error("gaussian_kde requires nonempty data and a positive finite bandwidth");
      }
      const double norm = 1.0 / (data.size() * h * std::sqrt(2 * kPi));
      double sum = 0.0;
      for (auto& xi : data) {
        double u = (x - xi) / h;
        sum += std::exp(-0.5 * u * u);
      }
      return norm * sum;
    }
    double DEEPIBP::silverman_bandwidth(const std::vector<double>& data) {
      double n = data.size();
      if (n < 2.0) return 0.0;
      double mean = std::accumulate(data.begin(), data.end(), 0.0) / n;
      double var = 0.0;
      for (auto& x : data) var += (x - mean) * (x - mean);
      var /= n;
      double sigma = std::sqrt(var);
      return 1.06 * sigma * std::pow(n, -0.2);
    }
    double DEEPIBP::normalized_kde(const std::vector<double>& data, double x,
      double h, double phi_min, double phi_max, int n_points) {
      double dx = (phi_max - phi_min) / n_points;
      double norm = 0.0;
      for (int i = 0; i <= n_points; i++) {
        double xi = phi_min + i * dx;
        norm += gaussian_kde(data, xi, h) * dx;
      }
      if (norm <= 1e-12) return 0.0;
      return gaussian_kde(data, x, h) / norm;
    }
    double DEEPIBP::compute_kl(const std::vector<double>& data, double phi_min, double phi_max, const std::string& mode, int n_points) {
      double interval_len = phi_max - phi_min;
      if (data.empty()) error("compute_kl requires at least one sample");
      if (!std::isfinite(interval_len) || interval_len <= 0.0) error("compute_kl requires phi_max > phi_min");
      if (n_points < 2) error("compute_kl requires at least two integration points");
      if (mode != "Q||P" && mode != "P||Q") error("compute_kl mode must be Q||P or P||Q");
      const double bandwidth_floor = interval_len / static_cast<double>(n_points);
      double h = std::max(silverman_bandwidth(data), bandwidth_floor);
      double q = 1.0 / interval_len;
      double dx = interval_len / n_points;
      double kl = 0.0;
      for (int i = 0; i <= n_points; i++) {
        double x = phi_min + i * dx;
        double p_x = normalized_kde(data, x, h, phi_min, phi_max);
        if (p_x <= 1e-14) continue;
        if (mode == "Q||P") {
          kl += q * std::log(q / p_x) * dx;
        }
        else if (mode == "P||Q") {
          kl += p_x * std::log(p_x / q) * dx;
        }
      }
      return kl;
    }
    double DEEPIBP::compute_kl_row(const std::vector<double>& data, double phi_min, double phi_max, const std::string& mode, int n_points) {
      return compute_kl(data, phi_min, phi_max, mode, n_points);
    }
    void DEEPIBP::computeKLFromFile(const std::string& cvfile, double phi_min, double phi_max, unsigned int cv_index) {
      unsigned int ncv = getNumberOfArguments();
      IFile ifile;
      ifile.open(cvfile);
      if (!ifile.isOpen()) {
        log.printf("[computeKLFromFile] Error: file is not open!\n");
        return;
      }
      std::vector<Value> tmpvalues;
      std::vector<std::vector<double>> allCVData(ncv);
      for (unsigned j = 0; j < ncv; ++j) {
        tmpvalues.emplace_back(this, getPntrToArgument(j)->getName(), false);
      }
      std::vector<double> cv(ncv);
      double time;
      while (scanonecv(&ifile, tmpvalues, time, cv)) {
        double timestep = getTimeStep();
        unsigned step = static_cast<unsigned>(time / timestep);
        if (step > initial_stride * num_sub) {
          double phi = cv[cv_index];
          if (phi >= phi_min && phi <= phi_max) {
            for (unsigned i = 0; i < ncv; ++i) {
              allCVData[i].push_back(cv[i]);
            }
          }
        }
      }
      for (unsigned i = 0; i < ncv; ++i) {
        const std::vector<double>& col = allCVData[i];
        if (!col.empty()) {
          double kl_qp = compute_kl(col, phi_min, phi_max, "Q||P");
          double kl_pq = compute_kl(col, phi_min, phi_max, "P||Q");
          log.printf("CV[%u] KL(Q||P) = %f, KL(P||Q) = %f\n", i, kl_qp, kl_pq);
        }
      }
    }
    double DEEPIBP::gaussian_kde_2d(const std::vector<std::array<double, 2>>& data, double x, double y, double hx, double hy) {
      if (data.empty() || !std::isfinite(hx) || !std::isfinite(hy) || hx <= 0.0 || hy <= 0.0) {
        error("gaussian_kde_2d requires nonempty data and positive finite bandwidths");
      }
      const double norm = 1.0 / (data.size() * hx * hy * 2 * kPi);
      double sum = 0.0;
      for (auto& s : data) {
        double ux = (x - s[0]) / hx;
        double uy = (y - s[1]) / hy;
        sum += std::exp(-0.5 * (ux * ux + uy * uy));
      }
      return norm * sum;
    }
    std::array<double, 2> DEEPIBP::silverman_bandwidth_2d(const std::vector<std::array<double, 2>>& data) {
      size_t n = data.size();
      double meanx = 0.0, meany = 0.0;
      for (auto& s : data) { meanx += s[0]; meany += s[1]; }
      meanx /= n; meany /= n;
      double varx = 0.0, vary = 0.0;
      for (auto& s : data) {
        varx += (s[0] - meanx) * (s[0] - meanx);
        vary += (s[1] - meany) * (s[1] - meany);
      }
      varx /= n; vary /= n;
      double sigx = std::sqrt(varx);
      double sigy = std::sqrt(vary);
      double hx = 1.06 * sigx * std::pow(n, -1.0 / 6.0);
      double hy = 1.06 * sigy * std::pow(n, -1.0 / 6.0);
      return { hx, hy };
    }
    double DEEPIBP::compute_kl_2d(const std::vector<std::array<double, 2>>& data,
      const std::array<double, 2>& mins,
      const std::array<double, 2>& maxs,
      const std::string& mode,
      int n_points) {
      if (data.empty()) error("compute_kl_2d requires at least one sample");
      if (n_points < 2) error("compute_kl_2d requires at least two integration points");
      if (mode != "Q||P" && mode != "P||Q") error("compute_kl_2d mode must be Q||P or P||Q");
      const double width_x = maxs[0] - mins[0];
      const double width_y = maxs[1] - mins[1];
      if (!std::isfinite(width_x) || !std::isfinite(width_y) || width_x <= 0.0 || width_y <= 0.0) {
        error("compute_kl_2d requires maxs greater than mins");
      }
      const double dx = width_x / n_points;
      const double dy = width_y / n_points;
      auto h = silverman_bandwidth_2d(data);
      const double bandwidth_floor_x = dx;
      const double bandwidth_floor_y = dy;
      h[0] = std::max(h[0], bandwidth_floor_x);
      h[1] = std::max(h[1], bandwidth_floor_y);
      double area = width_x * width_y;
      double q = 1.0 / area;
      std::vector<double> density;
      density.reserve(static_cast<std::size_t>(n_points) * n_points);
      double density_normalization = 0.0;
      for (int i = 0; i < n_points; ++i) {
        for (int j = 0; j < n_points; ++j) {
          const double x = mins[0] + (i + 0.5) * dx;
          const double y = mins[1] + (j + 0.5) * dy;
          const double value = gaussian_kde_2d(data, x, y, h[0], h[1]);
          density.push_back(value);
          density_normalization += value * dx * dy;
        }
      }
      if (!std::isfinite(density_normalization) || density_normalization <= 0.0) {
        error("compute_kl_2d could not normalize the KDE inside the requested box");
      }
      double kl = 0.0;
      for (double raw_density : density) {
        const double p_xy = raw_density / density_normalization;
        if (p_xy <= 1e-14) continue;
        if (mode == "Q||P") {
          kl += q * std::log(q / p_xy) * dx * dy;
        }
        else {
          kl += p_xy * std::log(p_xy / q) * dx * dy;
        }
      }
      return kl;
    }
    double DEEPIBP::computeKLwithKDE(const std::vector<std::vector<double>>& samples,
      const std::vector<double>& mins,
      const std::vector<double>& maxs,
      const std::string& mode,
      int n_points) {
      if (monitor_cvs_.size() == 1) {
        std::vector<double> oneD;
        for (auto& s : samples) oneD.push_back(s[0]);
        return compute_kl(oneD, mins[0], maxs[0], mode, n_points);
      }
      else if (monitor_cvs_.size() == 2) {
        std::vector<std::array<double, 2>> twoD;
        for (auto& s : samples) twoD.push_back({ s[0], s[1] });
        return compute_kl_2d(twoD, { mins[0],mins[1] }, { maxs[0],maxs[1] }, mode, n_points);
      }
      else {
        error("computeKLwithKDE only supports 1D/2D");
        return -1.0;
      }
    }
    void DEEPIBP::calculate() {
      const unsigned ncv = getNumberOfArguments();
      std::vector<double> cv(ncv);
      for (unsigned i = 0; i < ncv; ++i) cv[i] = getArgument(i);
      if (!guess && subnn) {
        log.printf("The subnn phase has now started.\n");
        if (kl_region && in_tall_phase_ && !waiting_unbiased_ && !neutral_network) {
          if (monitor_cvs_.size() == 1) {
            const int idx0 = monitor_cvs_[0];
            const double cv0 = cv[idx0];
            const bool has_lower0 = (idx0 < (int)lower_wall.size()) && std::isfinite(lower_wall[idx0]);
            const bool has_upper0 = (idx0 < (int)upper_wall.size()) && std::isfinite(upper_wall[idx0]);
            const bool box0_ok = has_lower0 && has_upper0;
            if (box0_ok) {
              const bool inside = (cv0 >= lower_wall[idx0] && cv0 <= upper_wall[idx0]);
              ++tall_block_step_count_;
              if (inside) ++tall_block_inside_count_;
              if (tall_block_step_count_ >= tall_check_interval_) {
                const double frac =
                  static_cast<double>(tall_block_inside_count_) /
                  static_cast<double>(tall_block_step_count_);
                const double current_threshold = tall_threshold_[tall_stage_index_];
                log.printf("[KL-REGION][TALL-1D] step=%u stage=%zu/%zu window_steps=%d inside=%d frac=%.3f thr=%.3f\n",
                  getStep(), tall_stage_index_ + 1, height_tall_.size(),
                  tall_block_step_count_, tall_block_inside_count_,
                  frac, current_threshold);
                if (frac >= current_threshold) {
                  const std::size_t finished_stage = tall_stage_index_;
                  ++tall_stage_index_;
                  if (tall_stage_index_ >= height_tall_.size()) {
                    in_tall_phase_ = false;
                    log.printf("[KL-REGION][TALL-1D] stage=%zu threshold reached; switch to SMALL packets (HEIGHT_SUB=%.6f, SIGMA_SUB).\n",
                      finished_stage + 1, height_sub);
                  }
                  else {
                    log.printf("[KL-REGION][TALL-1D] stage=%zu threshold reached; switch to TALL stage=%zu, height=%.6f sigma[0]=%.6f.\n",
                      finished_stage + 1, tall_stage_index_ + 1,
                      height_tall_[tall_stage_index_], sigma_tall_[tall_stage_index_][0]);
                  }
                }
                tall_block_step_count_ = 0;
                tall_block_inside_count_ = 0;
              }
            }
          }
          else if (monitor_cvs_.size() == 2) {
            const int idx0 = monitor_cvs_[0];
            const int idx1 = monitor_cvs_[1];
            const double cv0 = cv[idx0];
            const double cv1 = cv[idx1];
            const bool has_lower0 = (idx0 < (int)lower_wall.size()) && std::isfinite(lower_wall[idx0]);
            const bool has_upper0 = (idx0 < (int)upper_wall.size()) && std::isfinite(upper_wall[idx0]);
            const bool has_lower1 = (idx1 < (int)lower_wall.size()) && std::isfinite(lower_wall[idx1]);
            const bool has_upper1 = (idx1 < (int)upper_wall.size()) && std::isfinite(upper_wall[idx1]);
            const bool box0_ok = has_lower0 && has_upper0;
            const bool box1_ok = has_lower1 && has_upper1;
            if (box0_ok && box1_ok) {
              const bool inside0 = (cv0 >= lower_wall[idx0] && cv0 <= upper_wall[idx0]);
              const bool inside1 = (cv1 >= lower_wall[idx1] && cv1 <= upper_wall[idx1]);
              const bool inside = inside0 && inside1;
              ++tall_block_step_count_;
              if (inside) ++tall_block_inside_count_;
              if (tall_block_step_count_ >= tall_check_interval_) {
                const double frac =
                  static_cast<double>(tall_block_inside_count_) /
                  static_cast<double>(tall_block_step_count_);
                const double current_threshold = tall_threshold_[tall_stage_index_];
                log.printf("[KL-REGION][TALL-2D] step=%u stage=%zu/%zu window_steps=%d inside=%d frac=%.3f thr=%.3f\n",
                  getStep(), tall_stage_index_ + 1, height_tall_.size(),
                  tall_block_step_count_, tall_block_inside_count_,
                  frac, current_threshold);
                if (frac >= current_threshold) {
                  const std::size_t finished_stage = tall_stage_index_;
                  ++tall_stage_index_;
                  if (tall_stage_index_ >= height_tall_.size()) {
                    in_tall_phase_ = false;
                    log.printf("[KL-REGION][TALL-2D] stage=%zu threshold reached; switch to SMALL packets (HEIGHT_SUB=%.6f, SIGMA_SUB).\n",
                      finished_stage + 1, height_sub);
                  }
                  else {
                    log.printf("[KL-REGION][TALL-2D] stage=%zu threshold reached; switch to TALL stage=%zu, height=%.6f sigma[0]=%.6f.\n",
                      finished_stage + 1, tall_stage_index_ + 1,
                      height_tall_[tall_stage_index_], sigma_tall_[tall_stage_index_][0]);
                  }
                }
                tall_block_step_count_ = 0;
                tall_block_inside_count_ = 0;
              }
            }
          }
        }
        bool nowaddhill = false;
        if (getStep() % current_stride_ == 0 && populated_gaussian < num_sub && !isFirstStep_&&!neutral_network) {
          nowaddhill = true;
          populated_gaussian++;
        }
        else {
          nowaddhill = false;
          isFirstStep_ = false;
        }
        if (waiting_unbiased_) nowaddhill = false;
        std::vector<double> der(ncv, 0.0);
        double ene = getBiasAndDerivatives(cv, der);
        setBias(ene);
        for (unsigned i = 0; i < ncv; i++) setOutputForce(i, -der[i]);
        std::vector<double> thissigma = sigma_sub;
        bool multivariate = false;
        double height = height_sub;
        bool using_tall_hill = false;
        if (kl_region && (monitor_cvs_.size() == 1 || monitor_cvs_.size() == 2) && !waiting_unbiased_ && !neutral_network) {
          if (in_tall_phase_) {
            plumed_assert(tall_stage_index_ < height_tall_.size());
            plumed_assert(tall_stage_index_ < sigma_tall_.size());
            thissigma = sigma_tall_[tall_stage_index_];
            height = height_tall_[tall_stage_index_];
            using_tall_hill = true;
            log.printf("[TALL-PHASE] Adding TALL Gaussian hill: stage=%zu/%zu height=%.4f sigma[0]=%.4f ...\n",
              tall_stage_index_ + 1, height_tall_.size(), height, thissigma[0]);
          }
          else {
          }
        }
        if (nowaddhill) {
          log.printf("Now starting to add Gaussians;\n");
          Gaussian newhill = Gaussian(multivariate, height, cv, thissigma);
          addGaussian(newhill);
          writeGaussian(newhill, hillsOfile_);
          if (!using_tall_hill) hill_cv_at_add_.push_back(cv);
          if (kl_region && (monitor_cvs_.size() == 1 || monitor_cvs_.size() == 2)) {
            unsigned long N = static_cast<unsigned long>(populated_gaussian);
            updateDynamicKStatistics(cv, N, using_tall_hill);
          }
        }
          if (kl_region) {
            std::vector<double> current;
            current.reserve(monitor_cvs_.size());
            for (int idx : monitor_cvs_) current.push_back(getArgument(idx));
            cv_trace_.push_back(current);
            if (waiting_unbiased_) {
              plumed_assert(a_box_.size() == b_box_.size() && a_box_.size() == monitor_cvs_.size());
              for (size_t k = 0; k < a_box_.size(); ++k) plumed_assert(a_box_[k] <= b_box_[k] && std::isfinite(a_box_[k]) && std::isfinite(b_box_[k]));
              plumed_assert(min_region_.size() >= a_box_.size());
              if (a_box_.size() == 1) {
                log.printf("[DEBUG] unbiased entered: dim=%zu a=%.6f b=%.6f\n",trigger_dim_for_unbiased_, a_box_[trigger_dim_for_unbiased_], b_box_[trigger_dim_for_unbiased_]);
                if (getStep() >= unbiased_start_step_ + N_unbiased) {
                  std::vector<std::vector<double>> window_samples;
                  window_samples.reserve(cv_trace_.size() - unbiased_start_index_);
                  for (size_t i = unbiased_start_index_; i < cv_trace_.size(); ++i) {
                    const auto& pt = cv_trace_[i];
                    bool inside = true;
                    for (size_t k = 0; k < a_box_.size(); ++k) {
                      if (pt[k] < a_box_[k] || pt[k] > b_box_[k]) { inside = false; break; }
                    }
                    if (inside) window_samples.push_back(pt);
                  }
                  if (window_samples.size() < static_cast<size_t>(samples_min_)) {
                    log.printf(" Not enough unbiased samples in the frozen box (have=%zu, need>=%d).\n",window_samples.size(), samples_min_);
                    error("Unbiased window insufficient samples for KL.");
                  }
                  size_t dim = trigger_dim_for_unbiased_;
                  double kl = computeKLwithKDE(window_samples, a_box_, b_box_, "P||Q", 200);
                  int window_iter = 0;
                  log.printf(" KL check begin: dim=%zu box=[%.6f, %.6f] samples=%zu KL=%.6g thr=%.6g\n", trigger_dim_for_unbiased_, a_box_[trigger_dim_for_unbiased_], b_box_[trigger_dim_for_unbiased_], window_samples.size(), kl, kl_threshold_);
                  while (kl > kl_threshold_ && (b_box_[dim] - a_box_[dim]) > min_region_[dim]) {
                    const double sigma = sigma_sub[monitor_cvs_[dim]];
                    const double shrink = 0.5 * sigma;
                    const bool has_custom_for_dim =(!custom_thr_.empty()) && (static_cast<size_t>(dim) < custom_thr_.size());
                    if (has_custom_for_dim) {
                      const double anchor = custom_thr_[dim];
                      const int op = (static_cast<size_t>(dim) < custom_op_.size() ? custom_op_[dim] : +1);
                      if (op < 0) {
                        b_box_[dim] = anchor;
                        a_box_[dim] = std::min(a_box_[dim] + shrink, b_box_[dim]);
                      }
                      else {
                        a_box_[dim] = anchor;
                        b_box_[dim] = std::max(b_box_[dim] - shrink, a_box_[dim]);
                      }
                    }
                    else {
                      if (unbiased_anchor_fixed_) {
                        b_box_[dim] = std::max(b_box_[dim] - shrink, a_box_[dim]);
                      }
                      else {
                        if (trigger_upper_for_unbiased_) {
                          a_box_[dim] = std::min(a_box_[dim] + shrink, b_box_[dim]);
                        }
                        else {
                          b_box_[dim] = std::max(b_box_[dim] - shrink, a_box_[dim]);
                        }
                      }
                    }
                    if (a_box_.size() > 1) {
                      for (size_t d = 0; d < a_box_.size(); ++d) {
                        if (d == static_cast<size_t>(dim)) continue;
                        const bool has_custom_d =
                          (!custom_thr_.empty()) && (d < custom_thr_.size());
                        if (!has_custom_d) continue;
                        const double sigma_d = sigma_sub[monitor_cvs_[d]];
                        const double shrink_d = 0.5 * sigma_d;
                        const double anchor_d = custom_thr_[d];
                        const int op_d = (d < custom_op_.size() ? custom_op_[d] : +1);
                        if (op_d < 0) {
                          b_box_[d] = anchor_d;
                          a_box_[d] = std::min(a_box_[d] + shrink_d, b_box_[d]);
                        }
                        else {
                          a_box_[d] = anchor_d;
                          b_box_[d] = std::max(b_box_[d] - shrink_d, a_box_[d]);
                        }
                      }
                    }
                    window_samples.clear();
                    for (size_t i = unbiased_start_index_; i < cv_trace_.size(); ++i) {
                      const auto& pt = cv_trace_[i];
                      bool inside = true;
                      for (size_t k = 0; k < a_box_.size(); ++k) {
                        if (pt[k] < a_box_[k] || pt[k] > b_box_[k]) { inside = false; break; }
                      }
                      if (inside) window_samples.push_back(pt);
                    }
                    if (window_samples.size() < static_cast<size_t>(samples_min_)) {
                      log.printf("[KL-REGION] Shrinked box has too few samples (have=%zu). Stop shrinking.\n",
                        window_samples.size());
                      break;
                    }
                    kl = computeKLwithKDE(window_samples, a_box_, b_box_, "P||Q", 200);
                    ++window_iter;
                    log.printf("[KL-REGION] iter=%d dim=%zu box=[%.6f, %.6f] samples=%zu KL=%.6g thr=%.6g\n",
                      window_iter, trigger_dim_for_unbiased_,
                      a_box_[trigger_dim_for_unbiased_], b_box_[trigger_dim_for_unbiased_],
                      window_samples.size(), kl, kl_threshold_);
                  }
                  if (kl <= kl_threshold_) {
                    fes_filter_dim_ = static_cast<size_t>(monitor_cvs_[dim]);
                    const std::string tag = std::string(trigger_upper_for_unbiased_ ? "upper" : "lower") + std::to_string(dim);
                    writeFESSegmentGridWallCorrected(a_box_[dim], b_box_[dim], tag);
                    unbiased_anchor_fixed_ = false;
                    plumed.stop();
                    return;
                  }
                  else {
                    std::ostringstream __oss;
                    __oss.setf(std::ios::fixed); __oss.precision(6);
                    __oss << "No suitable interval from unbiased window: "<< "KL=" << kl << " > thr=" << kl_threshold_ << " box=[" << a_box_[dim] << "," << b_box_[dim] << "]" << " width=" << (b_box_[dim] - a_box_[dim])<< " (min=" << min_region_[dim] << ").";
                    log.printf("[KL-REGION][ERROR] %s\n", __oss.str().c_str());
                    unbiased_anchor_fixed_ = false;
                    error(__oss.str());
                  }
                }
                return;
              }
              else if (a_box_.size() == 2) {
                log.printf(" 2D window active ,box0=[%.6f, %.6f], box1=[%.6f, %.6f]\n", a_box_[0], b_box_[0], a_box_[1], b_box_[1]);
                if (getStep() >= unbiased_start_step_ + N_unbiased) {
                  std::vector<std::vector<double>> window_samples;
                  window_samples.reserve(cv_trace_.size() - unbiased_start_index_);
                  for (size_t i = unbiased_start_index_; i < cv_trace_.size(); ++i) {
                    const auto& pt = cv_trace_[i];
                    if (pt[0] >= a_box_[0] && pt[0] <= b_box_[0] &&
                      pt[1] >= a_box_[1] && pt[1] <= b_box_[1]) {
                      window_samples.push_back(pt);
                    }
                  }
                  log.printf("2D box=[%.6f,%.6f]x[%.6f,%.6f] collected_samples=%zu (from unbiased window)\n",a_box_[0], b_box_[0], a_box_[1], b_box_[1],(unsigned long)window_samples.size());
                  if (window_samples.size() < static_cast<size_t>(samples_min_)) {
                    log.printf("[KL-REGION] Not enough unbiased samples in 2D box (have=%zu, need>=%d).\n",window_samples.size(), samples_min_);
                    error("Unbiased 2D window insufficient samples for KL.");
                  }
                  computeKLRowsFromCustomThr();
                  computeKLColsFromCustomThr();
                  writeKL2DSelectedPointsToFiles();
                  unbiased_anchor_fixed_ = false;
                  plumed.stop();
                  return;
                }
              }
            }
            else {
              if (monitor_cvs_.size() == 1) {
                const int idx = monitor_cvs_[0];
                const double cv_val = current[0];
                const bool has_lower = (idx < (int)lower_wall.size()) && std::isfinite(lower_wall[idx]);
                const bool has_upper = (idx < (int)upper_wall.size()) && std::isfinite(upper_wall[idx]);
                const bool custom_mode = (!custom_thr_.empty()) && ((has_lower && has_upper) || (!has_lower && !has_upper));
                if (custom_mode) {
                  const double anchor = custom_thr_[0];
                  const int op = custom_op_[0];
                  const double thr_cmp = custom_anchor_to_thr(anchor, op, dcv_[0]);
                  if (!in_tall_phase_ && hit_custom_anchor(cv_val, anchor, op, dcv_[0], idx)) {
                    if (!in_custom_bin_[0]) {
                      ++custom_hit_count_[0];
                      in_custom_bin_[0] = true;
                      log.printf("[TRIGGER] step=%u time=%.6f dim=0 side=custom " "cv=%.6f anchor=%.6f thr=%.6f op=%s hits=%d/%d\n",getStep(), getTimeStep() * getStep(), cv_val, anchor, thr_cmp, (op >= 0 ? "GE" : "LE"), custom_hit_count_[0], hit_trigger_times_);
                    }
                  }
                  else {
                    in_custom_bin_[0] = false;
                  }
                  if (custom_hit_count_[0] >= hit_trigger_times_) {
                    double a, b; custom_range_from_anchor(anchor, op, dcv_[0], a, b);
                    a_box_.assign(1, a); b_box_.assign(1, b);
                    height_sub = 0.0;
                    log.printf("[DIAG][enter-unbiased][rank=%u/%u] step=%u source=CUSTOM1D center=", comm.Get_rank(), comm.Get_size(), getStep());
                    for (size_t di = 0; di < current.size(); ++di) log.printf("%s%.10f", (di==0?"":" "), current[di]);
                    log.printf(" box=[%.10f, %.10f] hits=%d trigger=%d\n", a, b, custom_hit_count_[0], hit_trigger_times_);
                    waiting_unbiased_ = true;
                    unbiased_start_step_ = getStep();
                    unbiased_start_index_ = cv_trace_.size();
                    trigger_dim_for_unbiased_ = 0;
                    trigger_upper_for_unbiased_ = (op >= 0);
                    std::fill(custom_hit_count_.begin(), custom_hit_count_.end(), 0);
                    std::fill(in_custom_bin_.begin(), in_custom_bin_.end(), false);
                    std::fill(lower_hit_count_.begin(), lower_hit_count_.end(), 0);
                    std::fill(upper_hit_count_.begin(), upper_hit_count_.end(), 0);
                    std::fill(in_lower_bin_.begin(), in_lower_bin_.end(), false);
                    std::fill(in_upper_bin_.begin(), in_upper_bin_.end(), false);
                    log.printf("[KL-REGION] Enter unbiased window (CUSTOM 1D): start=%u; box=[%.6f, %.6f] ""(anchor=%.6f op=%s dcv=%.6f thr=%.6f)\n", unbiased_start_step_, a_box_[0], b_box_[0], anchor, (op >= 0 ? "GE" : "LE"), dcv_[0], thr_cmp);
                    return;
                  }
                }
                else {
                  if (has_lower) {
                    const double thrL = wall_thr(lower_wall[idx], dcv_[0]);
                    if (!in_tall_phase_ && hit_with_sign(cv_val, lower_wall[idx], dcv_[0])) {
                      if (!in_lower_bin_[0]) {
                        ++lower_hit_count_[0]; in_lower_bin_[0] = true;
                        log.printf("[TRIGGER] step=%u time=%.6f dim=0 side=lower cv=%.6f thr=%.6f hits=%d/%d\n", getStep(), getTimeStep() * getStep(), cv_val, thrL, lower_hit_count_[0], hit_trigger_times_);
                      }
                      if (lower_hit_count_[0] >= hit_trigger_times_) {
                        handleIntervalAndMaybeStop(0, false);
                        return;
                      }
                    }
                    else {
                      in_lower_bin_[0] = false;
                    }
                  }
                  if (has_upper) {
                    const double thrU = wall_thr(upper_wall[idx], dcv_[0]);
                    if (!in_tall_phase_ && hit_with_sign(cv_val, upper_wall[idx], dcv_[0])) {
                      if (!in_upper_bin_[0]) {
                        ++upper_hit_count_[0]; in_upper_bin_[0] = true;
                        log.printf("[TRIGGER] step=%u time=%.6f dim=0 side=upper cv=%.6f thr=%.6f hits=%d/%d\n", getStep(), getTimeStep() * getStep(), cv_val, thrU, upper_hit_count_[0], hit_trigger_times_);
                      }
                      if (upper_hit_count_[0] >= hit_trigger_times_) {
                        handleIntervalAndMaybeStop(0, true);
                        return;
                      }
                    }
                    else {
                      in_upper_bin_[0] = false;
                    }
                  }
                }
              }
              else if (monitor_cvs_.size() == 2) {
                const int idx0 = monitor_cvs_[0], idx1 = monitor_cvs_[1];
                const double cv0 = current[0], cv1 = current[1];
                const bool has_lower0 = (idx0 < (int)lower_wall.size()) && std::isfinite(lower_wall[idx0]);
                const bool has_upper0 = (idx0 < (int)upper_wall.size()) && std::isfinite(upper_wall[idx0]);
                const bool has_lower1 = (idx1 < (int)lower_wall.size()) && std::isfinite(lower_wall[idx1]);
                const bool has_upper1 = (idx1 < (int)upper_wall.size()) && std::isfinite(upper_wall[idx1]);
                const bool custom0 = (!custom_thr_.empty()) && ((has_lower0 && has_upper0) || (!has_lower0 && !has_upper0));
                const bool custom1 = (!custom_thr_.empty()) && ((has_lower1 && has_upper1) || (!has_lower1 && !has_upper1));
                const bool cv0_in_wall =
                  (!has_lower0 || cv0 >= lower_wall[idx0]) &&
                  (!has_upper0 || cv0 <= upper_wall[idx0]);
                const bool cv1_in_wall =
                  (!has_lower1 || cv1 >= lower_wall[idx1]) &&
                  (!has_upper1 || cv1 <= upper_wall[idx1]);
                const bool inside_2d_wall = cv0_in_wall && cv1_in_wall;
                bool dim0_triggered = false, dim1_triggered = false;
                bool dim0_upper = false, dim1_upper = false;
                if (custom0) {
                  const double anchor0 = custom_thr_[0];
                  const int op0 = custom_op_[0];
                  const double thr0 = custom_anchor_to_thr(anchor0, op0, dcv_[0]);
                  if (!in_tall_phase_ &&
                    inside_2d_wall &&
                    hit_custom_anchor(cv0, anchor0, op0, dcv_[0], idx0)) {
                    if (!in_custom_bin_[0]) {
                      ++custom_hit_count_[0];
                      in_custom_bin_[0] = true;
                      log.printf( "[TRIGGER] step=%u time=%.6f dim=0 side=custom " "cv=%.6f anchor=%.6f thr=%.6f op=%s hits=%d/%d\n", getStep(), getTimeStep() * getStep(), cv0, anchor0, thr0, (op0 >= 0 ? "GE" : "LE"), custom_hit_count_[0], hit_trigger_times_);
                    }
                  }
                  else {
                    in_custom_bin_[0] = false;
                  }
                  dim0_triggered = (custom_hit_count_[0] >= hit_trigger_times_);
                }
                else {
                  if (has_lower0) {
                    const double thr0L = wall_thr(lower_wall[idx0], dcv_[0]);
                    if (!in_tall_phase_ && hit_with_sign(cv0, lower_wall[idx0], dcv_[0])) {
                      if (!in_lower_bin_[0]) {
                        ++lower_hit_count_[0];
                        in_lower_bin_[0] = true;
                        log.printf(
                          "[TRIGGER] step=%u time=%.6f dim=0 side=lower "
                          "cv=%.6f thr=%.6f hits=%d/%d\n",
                          getStep(), getTimeStep() * getStep(),
                          cv0, thr0L, lower_hit_count_[0], hit_trigger_times_);
                        if (lower_hit_count_[0] >= hit_trigger_times_) {
                          log.printf(
                            "[TRIGGER] threshold satisfied at step=%u time=%.6f "
                            "dim=0 side=lower hits=%d/%d\n",
                            getStep(), getTimeStep() * getStep(),
                            lower_hit_count_[0], hit_trigger_times_);
                        }
                      }
                    }
                    else {
                      in_lower_bin_[0] = false;
                    }
                  }
                  if (has_upper0) {
                    const double thr0U = wall_thr(upper_wall[idx0], dcv_[0]);
                    if (!in_tall_phase_ && hit_with_sign(cv0, upper_wall[idx0], dcv_[0])) {
                      if (!in_upper_bin_[0]) {
                        ++upper_hit_count_[0];
                        in_upper_bin_[0] = true;
                        log.printf(
                          "[TRIGGER] step=%u time=%.6f dim=0 side=upper "
                          "cv=%.6f thr=%.6f hits=%d/%d\n",
                          getStep(), getTimeStep() * getStep(),
                          cv0, thr0U, upper_hit_count_[0], hit_trigger_times_);
                        if (upper_hit_count_[0] >= hit_trigger_times_) {
                          log.printf(
                            "[TRIGGER] threshold satisfied at step=%u time=%.6f "
                            "dim=0 side=upper hits=%d/%d\n",
                            getStep(), getTimeStep() * getStep(),
                            upper_hit_count_[0], hit_trigger_times_);
                        }
                      }
                    }
                    else {
                      in_upper_bin_[0] = false;
                    }
                  }
                  dim0_triggered =
                    ((has_lower0 && lower_hit_count_[0] >= hit_trigger_times_) ||
                      (has_upper0 && upper_hit_count_[0] >= hit_trigger_times_));
                  dim0_upper = has_upper0 &&
                    (upper_hit_count_[0] >= hit_trigger_times_);
                }
                if (custom1) {
                  const double anchor1 = custom_thr_[1];
                  const int op1 = custom_op_[1];
                  const double thr1 = custom_anchor_to_thr(anchor1, op1, dcv_[1]);
                  if (!in_tall_phase_ &&
                    inside_2d_wall &&
                    hit_custom_anchor(cv1, anchor1, op1, dcv_[1], idx1)) {
                    if (!in_custom_bin_[1]) {
                      ++custom_hit_count_[1];
                      in_custom_bin_[1] = true;
                      log.printf(
                        "[TRIGGER] step=%u time=%.6f dim=1 side=custom "
                        "cv=%.6f anchor=%.6f thr=%.6f op=%s hits=%d/%d\n",
                        getStep(), getTimeStep() * getStep(),
                        cv1, anchor1, thr1, (op1 >= 0 ? "GE" : "LE"),
                        custom_hit_count_[1], hit_trigger_times_);
                    }
                  }
                  else {
                    in_custom_bin_[1] = false;
                  }
                  dim1_triggered = (custom_hit_count_[1] >= hit_trigger_times_);
                }
                else {
                  if (has_lower1) {
                    const double thr1L = wall_thr(lower_wall[idx1], dcv_[1]);
                    if (!in_tall_phase_ && hit_with_sign(cv1, lower_wall[idx1], dcv_[1])) {
                      if (!in_lower_bin_[1]) {
                        ++lower_hit_count_[1];
                        in_lower_bin_[1] = true;
                        log.printf(
                          "[TRIGGER] step=%u time=%.6f dim=1 side=lower "
                          "cv=%.6f thr=%.6f hits=%d/%d\n",
                          getStep(), getTimeStep() * getStep(),
                          cv1, thr1L, lower_hit_count_[1], hit_trigger_times_);
                        if (lower_hit_count_[1] >= hit_trigger_times_) {
                          log.printf(
                            "[TRIGGER] threshold satisfied at step=%u time=%.6f "
                            "dim=1 side=lower hits=%d/%d\n",
                            getStep(), getTimeStep() * getStep(),
                            lower_hit_count_[1], hit_trigger_times_);
                        }
                      }
                    }
                    else {
                      in_lower_bin_[1] = false;
                    }
                  }
                  if (has_upper1) {
                    const double thr1U = wall_thr(upper_wall[idx1], dcv_[1]);
                    if (!in_tall_phase_ && hit_with_sign(cv1, upper_wall[idx1], dcv_[1])) {
                      if (!in_upper_bin_[1]) {
                        ++upper_hit_count_[1];
                        in_upper_bin_[1] = true;
                        log.printf(
                          "[TRIGGER] step=%u time=%.6f dim=1 side=upper "
                          "cv=%.6f thr=%.6f hits=%d/%d\n",
                          getStep(), getTimeStep() * getStep(),
                          cv1, thr1U, upper_hit_count_[1], hit_trigger_times_);
                        if (upper_hit_count_[1] >= hit_trigger_times_) {
                          log.printf(
                            "[TRIGGER] threshold satisfied at step=%u time=%.6f "
                            "dim=1 side=upper hits=%d/%d\n",
                            getStep(), getTimeStep() * getStep(),
                            upper_hit_count_[1], hit_trigger_times_);
                        }
                      }
                    }
                    else {
                      in_upper_bin_[1] = false;
                    }
                  }
                  dim1_triggered =
                    ((has_lower1 && lower_hit_count_[1] >= hit_trigger_times_) ||
                      (has_upper1 && upper_hit_count_[1] >= hit_trigger_times_));
                  dim1_upper = has_upper1 &&
                    (upper_hit_count_[1] >= hit_trigger_times_);
                }
                if (dim0_triggered && dim1_triggered) {
                  const double anchor0 = custom0 ? custom_thr_[0] : 0.0;
                  const int op0 = custom0 ? custom_op_[0] : +1;
                  const double anchor1 = custom1 ? custom_thr_[1] : 0.0;
                  const int op1 = custom1 ? custom_op_[1] : +1;
                  a_box_.assign(2, 0.0);
                  b_box_.assign(2, 0.0);
                  if (custom0) {
                    double a0, b0;
                    custom_range_from_anchor(anchor0, op0, dcv_[0], a0, b0);
                    a_box_[0] = a0;
                    b_box_[0] = b0;
                  }
                  else {
                    const double W0 = dim0_upper ? upper_wall[idx0] : lower_wall[idx0];
                    const double T0 = W0 + dcv_[0];
                    a_box_[0] = std::min(W0, T0);
                    b_box_[0] = std::max(W0, T0);
                  }
                  if (custom1) {
                    double a1, b1;
                    custom_range_from_anchor(anchor1, op1, dcv_[1], a1, b1);
                    a_box_[1] = a1;
                    b_box_[1] = b1;
                  }
                  else {
                    const double W1 = dim1_upper ? upper_wall[idx1] : lower_wall[idx1];
                    const double T1 = W1 + dcv_[1];
                    a_box_[1] = std::min(W1, T1);
                    b_box_[1] = std::max(W1, T1);
                  }
                  height_sub = 0.0;
                  log.printf("[DIAG][enter-unbiased][rank=%u/%u] step=%u source=2D-trigger current=", comm.Get_rank(), comm.Get_size(), getStep());
                  for (size_t di = 0; di < current.size(); ++di) log.printf("%s%.10f", (di==0?"":" "), current[di]);
                  log.printf(" box0=[%.10f, %.10f] box1=[%.10f, %.10f]\n", a_box_[0], b_box_[0], a_box_[1], b_box_[1]);
                  waiting_unbiased_ = true;
                  unbiased_start_step_ = getStep();
                  unbiased_start_index_ = cv_trace_.size();
                  trigger_dim_for_unbiased_ = 0;
                  trigger_upper_for_unbiased_ = custom0 ? (op0 >= 0) : dim0_upper;
                  unbiased_anchor_fixed_ =
                    (custom0 && trigger_dim_for_unbiased_ == 0) ||
                    (custom1 && trigger_dim_for_unbiased_ == 1);
                  std::fill(custom_hit_count_.begin(), custom_hit_count_.end(), 0);
                  std::fill(in_custom_bin_.begin(), in_custom_bin_.end(), false);
                  std::fill(lower_hit_count_.begin(), lower_hit_count_.end(), 0);
                  std::fill(upper_hit_count_.begin(), upper_hit_count_.end(), 0);
                  std::fill(in_lower_bin_.begin(), in_lower_bin_.end(), false);
                  std::fill(in_upper_bin_.begin(), in_upper_bin_.end(), false);
                  const char* side0 = custom0 ? (op0 >= 0 ? "custom(GE)" : "custom(LE)"): (dim0_upper ? "upper" : "lower");
                  const char* side1 = custom1 ? (op1 >= 0 ? "custom(GE)" : "custom(LE)") : (dim1_upper ? "upper" : "lower");
                  const double thr0 = custom0 ? custom_anchor_to_thr(anchor0, op0, dcv_[0]) : (dim0_upper ? wall_thr(upper_wall[idx0], dcv_[0]) : wall_thr(lower_wall[idx0], dcv_[0]));
                  const double thr1 = custom1 ? custom_anchor_to_thr(anchor1, op1, dcv_[1]) : (dim1_upper ? wall_thr(upper_wall[idx1], dcv_[1]) : wall_thr(lower_wall[idx1], dcv_[1]));
                  log.printf(
                    "[KL-REGION] Enter unbiased window (2D) "
                    "dim0=%s thr0=%.6f hits(L/U/C)=(%d/%d/%d)  "
                    "dim1=%s thr1=%.6f hits(L/U/C)=(%d/%d/%d); "
                    "box0=[%.6f, %.6f] box1=[%.6f, %.6f]\n",
                    side0, thr0,
                    lower_hit_count_[0], upper_hit_count_[0], custom_hit_count_[0],
                    side1, thr1,
                    lower_hit_count_[1], upper_hit_count_[1], custom_hit_count_[1],
                    a_box_[0], b_box_[0], a_box_[1], b_box_[1]);
                  return;
                }
              }
            }
          }
          if (kl_region &&
            monitor_cvs_.size() == 2 &&
            !waiting_unbiased_ &&
            !in_tall_phase_ &&
            populated_gaussian == num_sub) {
            double D0 = std::fabs(dcv_[0]);
            double D1 = std::fabs(dcv_[1]);
            int K_used = 0;
            adjustKAndDWhenKNotReached2D(D0, D1, K_used);
            buildBoxFromCustomAnchorAndD(D0, D1);
            height_sub = 0.0;
            log.printf("[DIAG][enter-unbiased][rank=%u/%u] step=%u source=2D-fallback box0=[%.10f, %.10f] box1=[%.10f, %.10f] K_used=%d D0=%g D1=%g\n", comm.Get_rank(), comm.Get_size(), getStep(), a_box_[0], b_box_[0], a_box_[1], b_box_[1], K_used, D0, D1);
            waiting_unbiased_ = true;
            unbiased_start_step_ = getStep();
            unbiased_start_index_ = cv_trace_.size();
            trigger_dim_for_unbiased_ = 0;
            trigger_upper_for_unbiased_ = false;
            unbiased_anchor_fixed_ = false;
            std::fill(custom_hit_count_.begin(), custom_hit_count_.end(), 0);
            std::fill(in_custom_bin_.begin(), in_custom_bin_.end(), false);
            std::fill(lower_hit_count_.begin(), lower_hit_count_.end(), 0);
            std::fill(upper_hit_count_.begin(), upper_hit_count_.end(), 0);
            std::fill(in_lower_bin_.begin(), in_lower_bin_.end(), false);
            std::fill(in_upper_bin_.begin(), in_upper_bin_.end(), false);
            log.printf("[KL-REGION] NGauss=%u 已经用完，但没有达到 K 次触碰原始 D；"
              "在锚点到 D 的范围内重新搜索得到 D0'=%g, D1'=%g, 使用 K_used=%d 进入无偏阶段。\n",
              num_sub, D0, D1, K_used);
          }
          if (!kl_region && populated_gaussian == num_sub && !neutral_network) {
            log.printf("The unbiased phase is about to begin and use histogram ");
            if (!unbiased_phase_entered_) {
              current_stride_ = 30000;
              unbiased_phase_start_step_ = getStep();
              unbiased_phase_entered_ = true;
            }
            if (getStep() >= unbiased_phase_start_step_ + N_unbiased) {
              IFile ifilecv; ifilecv.link(*this);
              std::string fname = "cv.txt";
              if (!ifilecv.FileExist(fname)) {
                log.printf(" Error: cv.txt does not exist!\n");
                return;
              }
              ifilecv.open(fname);
              ifilecv.allowIgnoredFields();
              log.printf("The file cv.txt is already open\n");
              if (histogram) { readcv(ifilecv); }
              if (kde) { log.printf("The unbiased phase is about to begin and use kde "); processCV(ifilecv); }
              if (kl) {
                for (unsigned int i = 0; i < ncv; i++) {
                  double cv_min = lower_wall[i];
                  double cv_max = upper_wall[i];
                  computeKLFromFile("cv.txt", cv_min, cv_max, i);
                }
              }
            }
          }
          if (neutral_network) { runNeuralNetworkTrainingAndOutput(); }
        }
      else if (guess && !subnn) {
          std::vector<double> der(ncv, 0.0);
          double ene = 0.0;
          if (biasf_ != 1.0) ene = getBiasAndDerivatives(cv, der);
          setBias(ene);
          for (unsigned i = 0; i < ncv; ++i) setOutputForce(i, -der[i]);
          bool nowAddAHill;
          if (getStep() % current_stride_ == 0 && !isFirstStep_) {
            nowAddAHill = true;
          }
          else {
            nowAddAHill = false;
            isFirstStep_ = false;
          }
          if (nowAddAHill) {
            double height = height_sub;
            std::vector<double> thissigma = sigma_sub;
            bool multivariate = false;
            for (size_t di = 0; di < cv.size(); ++di) log.printf("%s%.10f", (di==0?"":" "), cv[di]);
            Gaussian newhill = Gaussian(multivariate, height, cv, thissigma);
            addGaussian(newhill);
            writeGaussian(newhill, hillsOfile_);
            populated_gaussian++;
            if (populated_gaussian == num_sub && !sumhills_called_) {
              log.printf("Reached target number of Gaussians (%u). Calling sumhills...\n", num_sub);
              sumhills();
              sumhills_called_ = true;
            }
          }
        }
      }
    void DEEPIBP::updateDynamicKStatistics(const std::vector<double>& cv,
      unsigned long N,
      bool this_is_tall) {
      if (!use_dynamic_K_) {
        return;
      }
      const std::size_t nmon = monitor_cvs_.size();
      if (nmon == 0) return;
      if (dcv_.size() < nmon) return;
      if (waiting_unbiased_) return;
      if (dynamic_K_fixed_) return;
      if (delta_frozen_.size() != nmon) delta_frozen_.assign(nmon, false);
      if (delta_raw_.size() != nmon) delta_raw_.assign(nmon, 0.0);
      if (delta_clamped_.size() != nmon) delta_clamped_.assign(nmon, 0.0);
      if (deltaN_over_N_.size() != nmon) deltaN_over_N_.assign(nmon, 0.0);
      if (this_is_tall) {
        tall_used_ = true;
        return;
      }
      ++small_N_for_dynK_;
      unsigned long effectiveN = small_N_for_dynK_;
      if (effectiveN == 0) return;
      if (!dyn_anchor_initialized_) {
        dyn_anchor_initialized_ = true;
        dyn_anchor_values_.assign(nmon, 0.0);
        for (std::size_t k = 0; k < nmon; ++k) {
          int arg_index = monitor_cvs_[k];
          dyn_anchor_values_[k] = cv[arg_index];
        }
      }
      std::vector<bool> was_hit = delta_frozen_;
      const bool have_trace = !cv_trace_.empty();
      for (std::size_t k = 0; k < nmon; ++k) {
        int arg_index = monitor_cvs_[k];
        double D = std::fabs(dcv_[k]);
        if (D <= 0.0) {
          deltaN_over_N_[k] = 0.0;
          continue;
        }
        double cv_now = cv[arg_index];
        double anchor = custom_thr_.empty() ? dyn_anchor_values_[k] : custom_thr_[k];
        double maxd = 0.0;
        if (have_trace) {
          for (const auto& sample : cv_trace_) {
            if (k >= sample.size()) continue;
            double cv_hist = sample[k];
            double d_hist = anchor_distance(arg_index, cv_hist, anchor);
            if (d_hist > maxd) maxd = d_hist;
          }
        }
        double d_now = anchor_distance(arg_index, cv_now, anchor);
        if (d_now > maxd) maxd = d_now;
        delta_raw_[k] = maxd;
        double delta = maxd;
        if (delta > D) delta = D;
        delta_clamped_[k] = delta;
        deltaN_over_N_[k] = delta / static_cast<double>(effectiveN);
        if (delta >= D - 1e-8) {
          delta_frozen_[k] = true;
        }
      }
      std::vector<int> just_hit;
      for (std::size_t k = 0; k < nmon; ++k) {
        if (!was_hit[k] && delta_frozen_[k]) {
          just_hit.push_back(static_cast<int>(k));
        }
      }
      int slow_dim = -1;
      double slow_r = 0.0;
      for (std::size_t k = 0; k < nmon; ++k) {
        double r = deltaN_over_N_[k];
        if (r <= 0.0) continue;
        if (slow_dim < 0 || r < slow_r) {
          slow_dim = static_cast<int>(k);
          slow_r = r;
        }
      }
      if (slow_dim < 0) {
        log.printf("[KL-REGION][DYN-K] step=%u N=%lu slow_dim=-1 (no candidate dim)\n",
          (unsigned int)getStep(), (unsigned long)effectiveN);
        return;
      }
      steep_dim_for_K_ = slow_dim;
      double x = 0.0;
      double K_real = 0.0;
      int K_cand = hit_trigger_times_;
      if (slow_r > 0.0) {
        x = static_cast<double>(effectiveN) / slow_r;
        if (tall_used_) K_real = 9.27073 + 0.0381505 * x;
        else K_real = 8.57149 + 0.00376659 * x;
        K_cand = static_cast<int>(std::ceil(K_real));
        if (K_cand < min_K_) K_cand = min_K_;
        dynamic_K_ = K_cand;
        hit_trigger_times_ = dynamic_K_;
        dynamic_x_fixed_ = x;
      }
      log.printf("[KL-REGION][DYN-K] step=%u N=%lu slow_dim=%d tall=%d "
        "x=%.6f K_formula=%.6f K_cand=%d K_fixed=%d fixed=%d\n",
        (unsigned int)getStep(), (unsigned long)effectiveN,
        slow_dim, (tall_used_ ? 1 : 0),
        x, K_real, K_cand, dynamic_K_, (dynamic_K_fixed_ ? 1 : 0));
      for (std::size_t k = 0; k < nmon; ++k) {
        double D = std::fabs(dcv_[k]);
        log.printf("[KL-REGION][DYN-K]   dim=%zu delta_raw=%.6f delta=%.6f "
          "deltaN/N=%.6f hitD_prev=%d hitD_now=%d D=%.6f\n",
          k,
          delta_raw_[k],
          delta_clamped_[k],
          deltaN_over_N_[k],
          (was_hit[k] ? 1 : 0),
          (delta_frozen_[k] ? 1 : 0),
          D);
      }
      bool slow_dim_just_hit = false;
      for (int idx : just_hit) {
        if (idx == slow_dim) { slow_dim_just_hit = true; break; }
      }
      if (!slow_dim_just_hit || slow_r <= 0.0) return;
      dynamic_K_ = K_cand;
      if (dynamic_K_ < min_K_) dynamic_K_ = min_K_;
      hit_trigger_times_ = dynamic_K_;
      dynamic_K_fixed_ = true;
      dynamic_x_fixed_ = x;
      double D_slow = std::fabs(dcv_[slow_dim]);
      double delta_s = delta_clamped_[static_cast<std::size_t>(slow_dim)];
      double r_s = deltaN_over_N_[static_cast<std::size_t>(slow_dim)];
      log.printf("[KL-REGION][DYN-K-FIX] step=%u N=%lu slow_dim=%d "
        "D=%.6f delta=%.6f deltaN/N=%.6f "
        "x=%.6f K_formula=%.6f K_fixed=%d tall=%d\n",
        (unsigned int)getStep(), (unsigned long)effectiveN,
        slow_dim,
        D_slow, delta_s, r_s,
        x, K_real, dynamic_K_, (tall_used_ ? 1 : 0));
    }
    void DEEPIBP::buildBoxFromCustomThr() {
      if (monitor_cvs_.size() < 2) return;
      const size_t d0 = monitor_cvs_[0];
      const size_t d1 = monitor_cvs_[1];
      if (a_box_.size() < 2) a_box_.assign(2, 0.0);
      if (b_box_.size() < 2) b_box_.assign(2, 0.0);
      a_box_[0] = lower_wall[d0];
      b_box_[0] = upper_wall[d0];
      a_box_[1] = lower_wall[d1];
      b_box_[1] = upper_wall[d1];
    }
    bool DEEPIBP::collectBandSamples2D(double x_min, double x_max,
      double y_min, double y_max,
      std::vector<double>& xs) const {
      xs.clear();
      if (cv_trace_.empty() || monitor_cvs_.size() < 2) return false;
      for (const auto& frame : cv_trace_) {
        if (frame.size() < 2) continue;
        double x = frame[0];
        double y = frame[1];
        if (x >= x_min && x <= x_max &&
          y >= y_min && y <= y_max) {
          xs.push_back(x);
        }
      }
      return !xs.empty();
    }
    bool DEEPIBP::collectBandSamples2D_Y(double x_min, double x_max,
      double y_min, double y_max,
      std::vector<double>& ys) const {
      ys.clear();
      if (cv_trace_.empty() || monitor_cvs_.size() < 2) return false;
      for (const auto& frame : cv_trace_) {
        if (frame.size() < 2) continue;
        double x = frame[0];
        double y = frame[1];
        if (x >= x_min && x <= x_max &&
          y >= y_min && y <= y_max) {
          ys.push_back(y);
        }
      }
      return !ys.empty();
    }
    void DEEPIBP::computeKLRowsFromCustomThr() {
      if (monitor_cvs_.size() < 2) return;
      if (a_box_.size() < 2 || b_box_.size() < 2) buildBoxFromCustomThr();
      if (custom_thr_.empty()) return;
      const int gridN = kl_gridN_;
      const double dy = (b_box_[1] - a_box_[1]) / (double)gridN;
      kl_row_ok_.assign(gridN + 1, 0);
      kl_row_L_.assign(gridN + 1, std::numeric_limits<double>::quiet_NaN());
      kl_row_R_.assign(gridN + 1, std::numeric_limits<double>::quiet_NaN());
      double x_anchor = custom_thr_[0];
      int x_dir = (custom_op_.empty() ? +1 : custom_op_[0]);
      for (int j = 0; j <= gridN; ++j) {
        double y0 = a_box_[1] + j * dy;
        double y1 = y0 + dy;
        double left = (x_dir >= 0) ? x_anchor : a_box_[0];
        double right = (x_dir >= 0) ? b_box_[0] : x_anchor;
        bool row_ok = false;
        while ((right - left) > min_region_[0]) {
          std::vector<double> xs;
          bool have = collectBandSamples2D(left, right, y0, y1, xs);
          size_t n = xs.size();
          if (!have || n < (size_t)samples_min_) {
            log.printf("[ROW] y=[%.6f,%.6f] FAIL: samples=%lu < %d in x=[%.6f,%.6f], shrink.\n",
              y0, y1, (unsigned long)n, samples_min_, left, right);
            double shrink = (b_box_[0] - a_box_[0]) / (double)gridN;
            if (x_dir >= 0) right -= shrink; else left += shrink;
            continue;
          }
          double klv = compute_kl_row(xs, left, right, "P||Q", 200);
          if (klv <= kl_threshold_) {
            row_ok = true;
            kl_row_L_[j] = left;
            kl_row_R_[j] = right;
            log.printf("[ROW] y=[%.6f,%.6f] OK KL=%.6g x=[%.6f,%.6f] samples=%lu\n",
              y0, y1, klv, left, right, (unsigned long)n);
            break;
          }
          else {
            log.printf("[ROW] y=[%.6f,%.6f] KL=%.6g > %.6g in x=[%.6f,%.6f] samples=%lu → shrink\n",
              y0, y1, klv, kl_threshold_, left, right, (unsigned long)n);
            double shrink = (b_box_[0] - a_box_[0]) / (double)gridN;
            if (x_dir >= 0) right -= shrink; else left += shrink;
          }
        }
        if (!row_ok) {
          log.printf("[ROW] y=[%.6f,%.6f] give up: width=%.6g <= min=%.6g\n",
            y0, y1, (right - left), min_region_[0]);
        }
        kl_row_ok_[j] = row_ok ? 1 : 0;
      }
    }
    void DEEPIBP::computeKLColsFromCustomThr() {
      if (monitor_cvs_.size() < 2) return;
      if (a_box_.size() < 2 || b_box_.size() < 2) buildBoxFromCustomThr();
      if (custom_thr_.size() < 2) return;
      const int gridN = kl_gridN_;
      const double dx = (b_box_[0] - a_box_[0]) / (double)gridN;
      kl_col_ok_.assign(gridN + 1, 0);
      kl_col_low_.assign(gridN + 1, std::numeric_limits<double>::quiet_NaN());
      kl_col_high_.assign(gridN + 1, std::numeric_limits<double>::quiet_NaN());
      double y_anchor = custom_thr_[1];
      int y_dir = (custom_op_.size() > 1 ? custom_op_[1] : +1);
      for (int i = 0; i <= gridN; ++i) {
        double x0 = a_box_[0] + i * dx;
        double x1 = x0 + dx;
        double low = (y_dir >= 0) ? y_anchor : a_box_[1];
        double high = (y_dir >= 0) ? b_box_[1] : y_anchor;
        bool col_ok = false;
        while ((high - low) > min_region_[1]) {
          std::vector<double> ys;
          bool have = collectBandSamples2D_Y(x0, x1, low, high, ys);
          size_t n = ys.size();
          if (!have || n < (size_t)samples_min_) {
            log.printf("[KL-REGION][COL] x=[%.6f,%.6f] FAIL: samples=%lu < %d in y=[%.6f,%.6f], shrink.\n",
              x0, x1, (unsigned long)n, samples_min_, low, high);
            double shrink = (b_box_[1] - a_box_[1]) / (double)gridN;
            if (y_dir >= 0) high -= shrink; else low += shrink;
            continue;
          }
          double klv = compute_kl_row(ys, low, high, "P||Q", 200);
          if (klv <= kl_threshold_) {
            col_ok = true;
            kl_col_low_[i] = low;
            kl_col_high_[i] = high;
            log.printf("[KL-REGION][COL] x=[%.6f,%.6f] OK KL=%.6g y=[%.6f,%.6f] samples=%lu\n",
              x0, x1, klv, low, high, (unsigned long)n);
            break;
          }
          else {
            log.printf("[KL-REGION][COL] x=[%.6f,%.6f] KL=%.6g > %.6g in y=[%.6f,%.6f] samples=%lu → shrink\n",
              x0, x1, klv, kl_threshold_, low, high, (unsigned long)n);
            double shrink = (b_box_[1] - a_box_[1]) / (double)gridN;
            if (y_dir >= 0) high -= shrink; else low += shrink;
          }
        }
        if (!col_ok) {
          log.printf("[KL-REGION][COL] x=[%.6f,%.6f] give up: height=%.6g <= min=%.6g\n",
            x0, x1, (high - low), min_region_[1]);
        }
        kl_col_ok_[i] = col_ok ? 1 : 0;
      }
    }
    void DEEPIBP::writeKL2DSelectedPointsToFiles() {
      auto* grid = dynamic_cast<Grid*>(BiasGrid_.get());
      if (!grid) error("BiasGrid_ is not of type Grid.");
      grid->scaleAllValuesAndDerivatives(-1.0);
      {
        OFile fes; fes.link(*this); fes.open(outhills);
        grid->setMinToZero();
        grid->setOutputFmt(fmt);
        grid->writeToFile(fes);
      }
      grid->scaleAllValuesAndDerivatives(-1.0);
      IFile fesIn; fesIn.link(*this);
      if (!fesIn.FileExist(outhills)) {
        error(std::string("[KL-REGION] Cannot find FES file ") + outhills);
      }
      fesIn.open(outhills);
      fesIn.allowIgnoredFields();
      std::string xfile = "filtered_fes_x_" + std::to_string(n_sub) + ".dat";
      std::string yfile = "filtered_fes_y_" + std::to_string(n_sub) + ".dat";
      OFile xout; xout.link(*this); xout.open(xfile);
      OFile yout; yout.link(*this); yout.open(yfile);
      const unsigned ncv = getNumberOfArguments();
      std::vector<double> cvs(ncv), ders(ncv);
      double bias = 0.0;
      const int gridN = kl_gridN_;
      const size_t d0 = monitor_cvs_[0];
      const size_t d1 = monitor_cvs_[1];
      const double dx = (b_box_[0] - a_box_[0]) / (double)gridN;
      const double dy = (b_box_[1] - a_box_[1]) / (double)gridN;
      while (true) {
        bool ok = true;
        for (unsigned i = 0; i < ncv; ++i)
          ok &= fesIn.scanField(getPntrToArgument(i)->getName(), cvs[i]);
        ok &= fesIn.scanField(getLabel() + ".bias", bias);
        for (unsigned i = 0; i < ncv; ++i)
          ok &= fesIn.scanField("der_" + getPntrToArgument(i)->getName(), ders[i]);
        if (!ok) break;
        fesIn.scanField();
        const double x = cvs[d0];
        const double y = cvs[d1];
        bool outside_wall = false;
        for (unsigned i = 0; i < ncv; ++i) {
          const double v = cvs[i];
          if (std::isfinite(lower_wall[i]) && v < lower_wall[i]) { outside_wall = true; break; }
          if (std::isfinite(upper_wall[i]) && v > upper_wall[i]) { outside_wall = true; break; }
        }
        if (outside_wall) continue;
        bool near_wall_inside = false;
        for (unsigned i = 0; i < ncv; ++i) {
          const double v = cvs[i];
          const double s = sigma_sub[i];
          if (std::isfinite(lower_wall[i]) &&
            std::fabs(v - lower_wall[i]) <= 0.5 * s) {
            near_wall_inside = true; break;
          }
          if (std::isfinite(upper_wall[i]) &&
            std::fabs(v - upper_wall[i]) <= 0.5 * s) {
            near_wall_inside = true; break;
          }
        }
        if (near_wall_inside) continue;
        int col = (int)((x - a_box_[0]) / dx + 1e-8);
        int row = (int)((y - a_box_[1]) / dy + 1e-8);
        if (col < 0 || col > gridN) continue;
        if (row < 0 || row > gridN) continue;
        bool pass_row = false;
        if (kl_row_ok_.size() == (size_t)gridN + 1 &&
          static_cast<size_t>(row) < kl_row_ok_.size() &&
          kl_row_ok_[row] == 1) {
          double L = kl_row_L_[row];
          double R = kl_row_R_[row];
          if (!std::isnan(L) && !std::isnan(R)) {
            if (x >= L && x <= R) {
              pass_row = true;
            }
          }
        }
        bool pass_col = false;
        if (kl_col_ok_.size() == (size_t)gridN + 1 &&
          static_cast<size_t>(col) < kl_col_ok_.size() &&
          kl_col_ok_[col] == 1) {
          double low = kl_col_low_[col];
          double high = kl_col_high_[col];
          if (!std::isnan(low) && !std::isnan(high)) {
            if (y >= low && y <= high) {
              pass_col = true;
            }
          }
        }
        if (pass_row) {
          for (unsigned i = 0; i < ncv; ++i)
            xout.printField(getPntrToArgument(i)->getName(), cvs[i]);
          xout.printField(getLabel() + ".bias", bias);
          for (unsigned i = 0; i < ncv; ++i)
            xout.printField("der_" + getPntrToArgument(i)->getName(), ders[i]);
          xout.printField();
        }
        if (pass_col) {
          for (unsigned i = 0; i < ncv; ++i)
            yout.printField(getPntrToArgument(i)->getName(), cvs[i]);
          yout.printField(getLabel() + ".bias", bias);
          for (unsigned i = 0; i < ncv; ++i)
            yout.printField("der_" + getPntrToArgument(i)->getName(), ders[i]);
          yout.printField();
        }
      }
      fesIn.close();
      xout.close();
      yout.close();
      log.printf("[KL-REGION] wrote KL-filtered (no-wall-correction) FES: %s  %s\n",
        xfile.c_str(), yfile.c_str());
    }
    void DEEPIBP::writeFESSegmentGridWallCorrected(double a, double b, const std::string& tag) {
      OFile hills; hills.link(*this);
      std::string corrected_fes = "filtered_fes_" + std::to_string(n_sub) + ".dat";
      auto* grid = dynamic_cast<Grid*>(BiasGrid_.get());
      if (!grid) error("BiasGrid_ is not of type Grid.");
      grid->scaleAllValuesAndDerivatives(-1.0);
      hills.open(outhills);
      grid->setMinToZero();
      grid->setOutputFmt(fmt);
      grid->writeToFile(hills);
      hills.close();
      OFile wallOut; wallOut.link(*this); wallOut.open(corrected_fes);
      IFile wallIn; wallIn.link(*this);
      if (!wallIn.FileExist(outhills)) { error("fes.dat missing"); }
      wallIn.open(outhills);
      wallIn.allowIgnoredFields();
      const unsigned ncv = getNumberOfArguments();
      if (fes_filter_dim_ >= ncv) {
        error("fes_filter_dim_ out of range. Did you forget to set it before calling writeFESSegmentGridWallCorrected?");
      }
      std::vector<double> wall_fields(ncv), wall_der(ncv);
      double bias = 0.0;
      while (true) {
        bool ok = true;
        for (unsigned i = 0; i < ncv; ++i)
          ok &= wallIn.scanField(getPntrToArgument(i)->getName(), wall_fields[i]);
        ok &= wallIn.scanField(getLabel() + ".bias", bias);
        for (unsigned i = 0; i < ncv; ++i)
          ok &= wallIn.scanField("der_" + getPntrToArgument(i)->getName(), wall_der[i]);
        if (!ok) break;
        wallIn.scanField();
        const double cvv = wall_fields[fes_filter_dim_];
        if (cvv < a || cvv > b) continue;
        bool near_any_wall = false;
        for (unsigned i = 0; i < ncv; ++i) {
          const double v = wall_fields[i];
          const double s = sigma_sub[i];
          if (std::isfinite(lower_wall[i])) {
            const double d_abs = std::fabs(v - lower_wall[i]);
            if (d_abs <= 0.5 * s) { near_any_wall = true; break; }
          }
          if (std::isfinite(upper_wall[i])) {
            const double d_abs = std::fabs(v - upper_wall[i]);
            if (d_abs <= 0.5 * s) { near_any_wall = true; break; }
          }
        }
        if (near_any_wall) continue;
        for (unsigned i = 0; i < ncv; ++i) {
          const double v = wall_fields[i];
          if (std::isfinite(lower_wall[i]) && v < lower_wall[i] && lower_wall[i] - v <= sigma_sub[i]) {
            const double d = lower_wall[i] - v;
            wall_der[i] -= kappa_l[i] * exp_l[i] * std::pow(d, exp_l[i] - 1);
            bias -= kappa_l[i] * std::pow(d, exp_l[i]);
          }
          else if (std::isfinite(upper_wall[i]) && v > upper_wall[i] && v - upper_wall[i] <= sigma_sub[i]) {
            const double d = v - upper_wall[i];
            wall_der[i] -= kappa_u[i] * exp_u[i] * std::pow(d, exp_u[i] - 1);
            bias -= kappa_u[i] * std::pow(d, exp_u[i]);
          }
        }
        for (unsigned i = 0; i < ncv; ++i)
          wallOut.printField(getPntrToArgument(i)->getName(), wall_fields[i]);
        wallOut.printField("file.free", bias);
        for (unsigned i = 0; i < ncv; ++i)
          wallOut.printField("der_" + getPntrToArgument(i)->getName(), wall_der[i]);
        wallOut.printField();
      }
      log.printf("Wall-corrected FES segment written: %s (range [%.6f, %.6f] on CV %zu)\n", corrected_fes.c_str(), a, b, fes_filter_dim_);
    }
    void DEEPIBP::sumhills() {
      if (!BiasGrid_) error("BiasGrid_ is null; initialize it before sumhills().");
      const unsigned stride_steps = static_cast<unsigned>(stride_hap);
      const bool write_final_even_with_stride = true;
      if (stride_steps == 0) {
        writeFESSnapshot_("hap.dat", true);
        return;
      }
      if (write_final_even_with_stride) {
        writeFESSnapshot_("hap.dat", true);
      }
      else {
        log.printf("[sumhills] STRIDEFES=%u and online snapshots already written in addGaussian(). "
          "No extra files are produced. Set WRITE_FINAL to write fes.dat here.\n", stride_steps);
      }
    }
    void DEEPIBP::handleIntervalAndMaybeStop(size_t dim, bool upper_side) {
      const int idx = monitor_cvs_.at(dim);
      const bool has_lower = (idx < (int)lower_wall.size()) && std::isfinite(lower_wall[idx]);
      const bool has_upper = (idx < (int)upper_wall.size()) && std::isfinite(upper_wall[idx]);
      const bool is_custom_dim = (!custom_thr_.empty()) && ((has_lower && has_upper) || (!has_lower && !has_upper));
      if (is_custom_dim) {
        error("handleIntervalAndMaybeStop() must not be called for CUSTOM mode; build a_box_/b_box_ at the call site.");
      }
      const size_t ndim = monitor_cvs_.size();
      a_box_.assign(ndim, 0.0);
      b_box_.assign(ndim, 0.0);
      for (size_t k = 0; k < ndim; ++k) {
        const int kidx = monitor_cvs_.at(k);
        if (k == dim) {
          plumed_assert(upper_side ? std::isfinite(upper_wall[kidx]) : std::isfinite(lower_wall[kidx]));
          const double W = upper_side ? upper_wall[kidx] : lower_wall[kidx];
          const double T = W + dcv_.at(k);
          a_box_[k] = std::min(W, T);
          b_box_[k] = std::max(W, T);
        }
        else {
          const double center = getArgument(kidx);
          a_box_[k] = center - tangential_half_[k];
          b_box_[k] = center + tangential_half_[k];
        }
      }
      height_sub = 0.0;
      log.printf("[DIAG][enter-unbiased][rank=%u/%u] step=%u source=handleInterval dim=%zu side=%s box=[%.10f, %.10f]\n", comm.Get_rank(), comm.Get_size(), getStep(), dim, upper_side ? "upper" : "lower", a_box_[dim], b_box_[dim]);
      waiting_unbiased_ = true;
      unbiased_start_step_ = getStep();
      unbiased_start_index_ = cv_trace_.size();
      trigger_dim_for_unbiased_ = dim;
      trigger_upper_for_unbiased_ = upper_side;
      log.printf("[KL-REGION] Enter unbiased window: dim=%zu side=%s start=%u time=%.6f box=[%.6f, %.6f]\n",
        dim, upper_side ? "upper" : "lower",
        unbiased_start_step_, getTimeStep() * getStep(),
        a_box_[dim], b_box_[dim]);
      std::fill(lower_hit_count_.begin(), lower_hit_count_.end(), 0);
      std::fill(upper_hit_count_.begin(), upper_hit_count_.end(), 0);
      std::fill(in_lower_bin_.begin(), in_lower_bin_.end(), false);
      std::fill(in_upper_bin_.begin(), in_upper_bin_.end(), false);
    }
    void DEEPIBP::readBackTwoDFourier(const std::string& filename, int nx, int ny, double x_min, double dx, double y_min, double dy, bool phi_is_x, std::vector<double>& dst) {
      IFile fit; fit.link(*this).open(filename);
      dst.assign(nx * ny, 0.0);
      auto clamp = [](int a, int lo, int hi) { return (a < lo ? lo : (a > hi ? hi : a)); };
      double phi, psi, v;
      while (true) {
        bool ok = true;
        ok &= fit.scanField("phi", phi);
        ok &= fit.scanField("psi", psi);
        ok &= fit.scanField("fitted_value", v);
        if (!ok) break;
        fit.scanField();
        int ix, iy;
        if (phi_is_x) {
          ix = (int)std::llround((phi - x_min) / dx);
          iy = (int)std::llround((psi - y_min) / dy);
        }
        else {
          iy = (int)std::llround((phi - y_min) / dy);
          ix = (int)std::llround((psi - x_min) / dx);
        }
        ix = clamp(ix, 0, nx - 1);
        iy = clamp(iy, 0, ny - 1);
        dst[iy * nx + ix] = v;
      }
    }
    std::vector<double> DEEPIBP::savgolFilter(const std::vector<double>& y, int window, int polyorder) {
      const int n = static_cast<int>(y.size());
      if (n == 0) return {};
      if (window < 1) throw std::invalid_argument("window length must be >= 1");
      if (window > n) window = (n % 2 == 1 ? n : n - 1);
      if (window % 2 == 0) --window;
      if (window < 1) window = 1;
      if (polyorder < 0 || polyorder >= window) {
        throw std::invalid_argument("polyorder must satisfy 0 <= polyorder < window");
      }
      const int half = window / 2;
      std::vector<double> result(n, 0.0);
      for (int i = 0; i < n; ++i) {
        int j0 = i - half;
        int j1 = i + half;
        if (j0 < 0) { j1 += -j0; j0 = 0; }
        if (j1 >= n) { j0 -= (j1 - (n - 1)); j1 = n - 1; }
        const int m = j1 - j0 + 1;
        Eigen::MatrixXd A(m, polyorder + 1);
        Eigen::VectorXd b(m);
        for (int r = 0; r < m; ++r) {
          const int j = j0 + r;
          const double x = static_cast<double>(j - i);
          double xp = 1.0;
          for (int p = 0; p <= polyorder; ++p) {
            A(r, p) = xp;
            xp *= x;
          }
          b(r) = y[j];
        }
        Eigen::VectorXd coeffs =
          (A.transpose() * A).ldlt().solve(A.transpose() * b);
        result[i] = coeffs(0);
      }
      return result;
    }
    void DEEPIBP::runNeuralNetworkTrainingAndOutput() {
      unsigned ncv = getNumberOfArguments();
      if (!neutral_network) return;
      if (ncv == 1) {
        OFile integratedFes;
        integratedFes.link(*this);
        integratedFes.open(intergrated_fes);
        for (unsigned i = 1; i <= n_sub; ++i) {
          std::string input_fes = "filtered_fes_" + std::to_string(i) + ".dat";
          IFile inFile;
          inFile.link(*this);
          if (!inFile.FileExist(input_fes)) {
            log.printf("WARNING: Could not open file %s. Skipping.\n", input_fes.c_str());
            continue;
          }
          inFile.open(input_fes);
          inFile.allowIgnoredFields();
          std::vector<double> cv_vals(ncv), der_vals(ncv);
          while (true) {
            bool ok = true;
            for (unsigned j = 0; j < ncv; ++j) {
              ok &= inFile.scanField(getPntrToArgument(j)->getName(), cv_vals[j]);
            }
            for (unsigned j = 0; j < ncv; ++j) {
              ok &= inFile.scanField("der_" + getPntrToArgument(j)->getName(), der_vals[j]);
            }
            if (!ok) break;
            inFile.scanField();
            for (unsigned j = 0; j < ncv; ++j) {
              integratedFes.printField(getPntrToArgument(j)->getName(), cv_vals[j]);
            }
            for (unsigned j = 0; j < ncv; ++j) {
              integratedFes.printField("der_" + getPntrToArgument(j)->getName(), der_vals[j]);
            }
            integratedFes.printField();
          }
          inFile.close();
        }
        integratedFes.close();
        IFile trainFile;
        trainFile.link(*this);
        trainFile.open(intergrated_fes);
        trainFile.allowIgnoredFields();
        std::vector<torch::Tensor> input_tensors, target_tensors;
        std::vector<double> cv_tmp(ncv), der_tmp(ncv);
        while (true) {
          bool ok = true;
          for (unsigned j = 0; j < ncv; ++j) {
            ok &= trainFile.scanField(getPntrToArgument(j)->getName(), cv_tmp[j]);
          }
          for (unsigned j = 0; j < ncv; ++j) {
            ok &= trainFile.scanField("der_" + getPntrToArgument(j)->getName(), der_tmp[j]);
          }
          if (!ok) break;
          trainFile.scanField();
          const auto float_options = torch::TensorOptions().dtype(torch::kFloat32);
          input_tensors.push_back(torch::tensor(cv_tmp, float_options).view({ 1, ncv }));
          target_tensors.push_back(torch::tensor(der_tmp, float_options).view({ 1, ncv }));
        }
        if (input_tensors.empty()) {
          error("No training data found in intergrated_fes for 1D NN.");
        }
        torch::Tensor inputs = torch::cat(input_tensors);
        torch::Tensor targets = torch::cat(target_tensors);
        if (inputs.isnan().any().item<bool>() || inputs.isinf().any().item<bool>()) {
          error("ERROR: Inputs contain NaN or inf values!");
        }
        if (targets.isnan().any().item<bool>() || targets.isinf().any().item<bool>()) {
          error("ERROR: Targets contain NaN or inf values!");
        }
        std::vector<std::string> min_vals(ncv), max_vals(ncv);
        for (unsigned i = 0; i < ncv; ++i) {
          min_vals[i] = std::to_string(gmin[i]);
          max_vals[i] = std::to_string(gmax[i]);
        }
        {
          int N_grid = 1000;
          auto optimizer = makeOptimizer(nn_model);
          for (int i = 0; i < epochs; ++i) {
            optimizer->zero_grad();
            auto outputs = nn_model->forward(inputs);
            auto loss = nn_model->loss(outputs, targets);
            loss.backward();
            optimizer->step();
            if (i % 100 == 0 || i == epochs - 1) {
              log.printf("Epoch [%d/%d], Loss: %.6f\n",
                i + 1, epochs, loss.item<double>());
              auto inputs_cpu = inputs.squeeze().to(torch::kCPU);
              auto outputs_cpu = outputs.squeeze().to(torch::kCPU);
              auto targets_cpu = targets.squeeze().to(torch::kCPU);
              for (int j = 0; j < std::min<int64_t>(5, inputs_cpu.size(0)); ++j) {
                log.printf("  Input[%d]: %.6f | Prediction: %.6f | Target: %.6f\n",
                  j,
                  inputs_cpu.index({ j }).item<float>(),
                  outputs_cpu.index({ j }).item<float>(),
                  targets_cpu.index({ j }).item<float>());
              }
            }
          }
        }
        OFile pointOfile_;
        pointOfile_.link(*this);
        pointOfile_.open("POINT");
        if (!pointOfile_.isOpen()) error("Cannot write POINT file");
        pointOfile_.fmtField(fmt);
        const int N_grid = point_grid;
        const double input_min = gmin[0];
        const double input_max = gmax[0];
        std::vector<double> x_vals(N_grid);
        for (int i = 0; i < N_grid; ++i) {
          x_vals[i] = input_min + i * (input_max - input_min) / (N_grid - 1);
        }
        std::vector<double> y_vals(N_grid);
        for (int i = 0; i < N_grid; ++i) {
          torch::Tensor x_tensor = torch::tensor({ { x_vals[i] } }, torch::kFloat);
          torch::Tensor y_tensor = nn_model->forward(x_tensor);
          y_vals[i] = y_tensor[0][0].item<double>();
        }
        std::vector<std::pair<double, double>> xy_pairs;
        xy_pairs.reserve(N_grid);
        for (int i = 0; i < N_grid; ++i) xy_pairs.emplace_back(x_vals[i], y_vals[i]);
        std::sort(xy_pairs.begin(), xy_pairs.end());
        for (int i = 0; i < N_grid; ++i) {
          x_vals[i] = xy_pairs[i].first;
          y_vals[i] = xy_pairs[i].second;
        }
        std::vector<double> y_smoothed(N_grid, 0.0);
        if (periodic[0]) {
          std::vector<double> y_filtered = savgolFilter(y_vals, 9, 3);
          {
            OFile tmpfile;
            tmpfile.link(*this);
            tmpfile.open("tmp_nn_output.dat");
            for (int i = 0; i < N_grid; ++i) {
              tmpfile.printField("phi", x_vals[i])
                .printField("value", y_filtered[i])
                .printField();
            }
            tmpfile.flush();
            tmpfile.close();
          }
          {
            IFile infile;
            infile.link(*this);
            infile.open("tmp_nn_output.dat");
            OFile outfile;
            outfile.link(*this);
            outfile.open("nn_fourier_fit.dat");
            fitOneDimFourier(infile, outfile, num_terms, gmin[0], gmax[0], N_grid);
            outfile.flush();
            outfile.close();
          }
          {
            IFile fitfile;
            fitfile.link(*this);
            fitfile.open("nn_fourier_fit.dat");
            double x_fit = 0.0, y_fit = 0.0;
            int count = 0;
            while (count < N_grid && fitfile.scanField("x", x_fit)) {
              fitfile.scanField("y_fit", y_fit);
              y_smoothed[count++] = y_fit;
              fitfile.scanField();
            }
            if (count != N_grid) {
              error("nn_fourier_fit.dat lines read != N_grid; "
                "check flush/close and fitOneDimFourier output grid.");
            }
          }
        }
        else {
          int win = 9;
          if (win > N_grid) win = (N_grid % 2 == 1 ? N_grid : N_grid - 1);
          if (win < 3) win = 3;
          int poly = std::min(3, win - 1);
          if (poly < 0) poly = 0;
          y_smoothed = savgolFilter(y_vals, win, poly);
        }
        if (static_cast<int>(y_smoothed.size()) != N_grid) {
          error("Size mismatch: y_smoothed vs N_grid in 1D pipeline.");
        }
        std::vector<double> integral_y(N_grid, 0.0);
        for (int j = 1; j < N_grid; ++j) {
          const double dx = x_vals[j] - x_vals[j - 1];
          integral_y[j] = integral_y[j - 1] + 0.5 * dx * (y_smoothed[j] + y_smoothed[j - 1]);
        }
        for (int j = 0; j < N_grid; ++j) {
          pointOfile_.printField("bias", -integral_y[j]);
          pointOfile_.printField(getPntrToArgument(0)->getName(), x_vals[j]);
          pointOfile_.printField("der_" + getPntrToArgument(0)->getName(), -y_smoothed[j]);
          pointOfile_.printField();
        }
        pointOfile_.flush();
        log.printf("Successfully wrote POINT file from neural network on continuous grid.\n");
        return;
      }
      if (ncv == 2) {
        std::vector<torch::Tensor> inputs_cv1, targets_cv1;
        std::vector<torch::Tensor> inputs_cv2, targets_cv2;
        std::vector<double> cv_tmp(ncv), der_tmp(ncv);
        for (unsigned i = 1; i <= n_sub; ++i) {
          {
            std::string file_x = "filtered_fes_x_" + std::to_string(i) + ".dat";
            IFile inFile;
            inFile.link(*this);
            if (!inFile.FileExist(file_x)) {
              log.printf("WARNING: Could not open file %s. Skipping X.\n", file_x.c_str());
            }
            else {
              inFile.open(file_x);
              inFile.allowIgnoredFields();
              while (true) {
                bool ok = true;
                for (unsigned j = 0; j < ncv; ++j) {
                  ok &= inFile.scanField(getPntrToArgument(j)->getName(), cv_tmp[j]);
                }
                for (unsigned j = 0; j < ncv; ++j) {
                  ok &= inFile.scanField("der_" + getPntrToArgument(j)->getName(), der_tmp[j]);
                }
                if (!ok) break;
                inFile.scanField();
                const auto float_options = torch::TensorOptions().dtype(torch::kFloat32);
                inputs_cv1.push_back(torch::tensor(cv_tmp, float_options).view({ 1, ncv }));
                double d1 = der_tmp[0];
                targets_cv1.push_back(torch::tensor(std::vector<double>{ d1 }, float_options).view({ 1,1 }));
              }
              inFile.close();
            }
          }
          {
            std::string file_y = "filtered_fes_y_" + std::to_string(i) + ".dat";
            IFile inFile;
            inFile.link(*this);
            if (!inFile.FileExist(file_y)) {
              log.printf("WARNING: Could not open file %s. Skipping Y.\n", file_y.c_str());
            }
            else {
              inFile.open(file_y);
              inFile.allowIgnoredFields();
              while (true) {
                bool ok = true;
                for (unsigned j = 0; j < ncv; ++j) {
                  ok &= inFile.scanField(getPntrToArgument(j)->getName(), cv_tmp[j]);
                }
                for (unsigned j = 0; j < ncv; ++j) {
                  ok &= inFile.scanField("der_" + getPntrToArgument(j)->getName(), der_tmp[j]);
                }
                if (!ok) break;
                inFile.scanField();
                const auto float_options = torch::TensorOptions().dtype(torch::kFloat32);
                inputs_cv2.push_back(torch::tensor(cv_tmp, float_options).view({ 1, ncv }));
                double d2 = der_tmp[1];
                targets_cv2.push_back(torch::tensor(std::vector<double>{ d2 }, float_options).view({ 1,1 }));
              }
              inFile.close();
            }
          }
        }
        if (inputs_cv1.empty() || inputs_cv2.empty()) {
          error("No training data found for 2D NN (x or y).");
        }
        torch::Tensor inputs_x = torch::cat(inputs_cv1);
        torch::Tensor targets_x = torch::cat(targets_cv1);
        torch::Tensor inputs_y = torch::cat(inputs_cv2);
        torch::Tensor targets_y = torch::cat(targets_cv2);
        if (inputs_x.isnan().any().item<bool>() || inputs_x.isinf().any().item<bool>() ||
          targets_x.isnan().any().item<bool>() || targets_x.isinf().any().item<bool>() ||
          inputs_y.isnan().any().item<bool>() || inputs_y.isinf().any().item<bool>() ||
          targets_y.isnan().any().item<bool>() || targets_y.isinf().any().item<bool>()) {
          error("ERROR: 2D NN training data contain NaN or inf values!");
        }
        {
          auto optimizer = makeOptimizer(nn_model_cv1);
          for (int e = 0; e < epochs; ++e) {
            optimizer->zero_grad();
            auto out = nn_model_cv1->forward(inputs_x);
            auto loss = nn_model_cv1->loss(out, targets_x);
            loss.backward();
            optimizer->step();
            if (e % 100 == 0 || e == epochs - 1) {
              log.printf("[cv1] Epoch [%d/%d], Loss: %.6f\n", e + 1, epochs, loss.item<double>());
            }
          }
        }
        {
          auto optimizer = makeOptimizer(nn_model_cv2);
          for (int e = 0; e < epochs; ++e) {
            optimizer->zero_grad();
            auto out = nn_model_cv2->forward(inputs_y);
            auto loss = nn_model_cv2->loss(out, targets_y);
            loss.backward();
            optimizer->step();
            if (e % 100 == 0 || e == epochs - 1) {
              log.printf("[cv2] Epoch [%d/%d], Loss: %.6f\n", e + 1, epochs, loss.item<double>());
            }
          }
        }
        OFile pointOfile_;
        pointOfile_.link(*this);
        pointOfile_.open("POINT");
        if (!pointOfile_.isOpen()) error("Cannot write POINT file");
        pointOfile_.fmtField(fmt);
        const int nx = point_grid;
        const int ny = point_grid;
        const int N_grid = point_grid;
        const double x_min = gmin[0], x_max = gmax[0];
        const double y_min = gmin[1], y_max = gmax[1];
        const double dx = (x_max - x_min) / (nx - 1);
        const double dy = (y_max - y_min) / (ny - 1);
        auto lin_idx = [nx](int iy, int ix) { return iy * nx + ix; };
        std::vector<double> cv1_vals; cv1_vals.reserve(nx * ny);
        std::vector<double> cv2_vals; cv2_vals.reserve(nx * ny);
        std::vector<torch::Tensor> input_list; input_list.reserve(nx * ny);
        for (int iy = 0; iy < ny; ++iy) {
          const double y = y_min + iy * dy;
          for (int ix = 0; ix < nx; ++ix) {
            const double x = x_min + ix * dx;
            cv1_vals.push_back(x);
            cv2_vals.push_back(y);
            input_list.push_back(torch::tensor({ x, y }, torch::kFloat32));
          }
        }
        torch::Tensor grid_inputs = torch::stack(input_list).to(torch::kFloat32);
        torch::Tensor out1 = nn_model_cv1->forward(grid_inputs);
        torch::Tensor out2 = nn_model_cv2->forward(grid_inputs);
        std::vector<double> der1_vals; der1_vals.reserve(nx * ny);
        std::vector<double> der2_vals; der2_vals.reserve(nx * ny);
        for (int k = 0; k < grid_inputs.size(0); ++k) {
          der1_vals.push_back(out1[k][0].item<double>());
          der2_vals.push_back(out2[k][0].item<double>());
        }
        int win = 9;
        if (win > nx) win = (nx % 2 == 1 ? nx : nx - 1);
        if (win < 3) win = 3;
        int poly = std::min(3, win - 2);
        std::vector<double> der1_smoothed = der1_vals;
        std::vector<double> der2_smoothed = der2_vals;
        {
          std::vector<double> row(nx), row_s(nx);
          for (int iy = 0; iy < ny; ++iy) {
            for (int ix = 0; ix < nx; ++ix) row[ix] = der1_smoothed[lin_idx(iy, ix)];
            row_s = savgolFilter(row, win, poly);
            for (int ix = 0; ix < nx; ++ix) der1_smoothed[lin_idx(iy, ix)] = row_s[ix];
            for (int ix = 0; ix < nx; ++ix) row[ix] = der2_smoothed[lin_idx(iy, ix)];
            row_s = savgolFilter(row, win, poly);
            for (int ix = 0; ix < nx; ++ix) der2_smoothed[lin_idx(iy, ix)] = row_s[ix];
          }
        }
        if (getPntrToArgument(0)->isPeriodic()) {
          OFile t1; t1.link(*this).open("tmp_cv1_fourier_in.dat");
          for (size_t k = 0; k < cv1_vals.size(); ++k) {
            t1.printField("phi", cv1_vals[k])
              .printField("psi", cv2_vals[k])
              .printField("value", der1_smoothed[k])
              .printField();
          }
          t1.flush();
          IFile fin1; fin1.link(*this).open("tmp_cv1_fourier_in.dat");
          OFile fout1; fout1.link(*this).open("cv1_fourier_fit.dat");
          twodfourier(fin1, fout1, 4);
          readBackTwoDFourier("cv1_fourier_fit.dat", nx, ny, x_min, dx, y_min, dy, true, der1_smoothed);
        }
        if (getPntrToArgument(1)->isPeriodic()) {
          OFile t2; t2.link(*this).open("tmp_cv2_fourier_in.dat");
          for (size_t k = 0; k < cv1_vals.size(); ++k) {
            t2.printField("phi", cv2_vals[k])
              .printField("psi", cv1_vals[k])
              .printField("value", der2_smoothed[k])
              .printField();
          }
          t2.flush();
          IFile fin2; fin2.link(*this).open("tmp_cv2_fourier_in.dat");
          OFile fout2; fout2.link(*this).open("cv2_fourier_fit.dat");
          twodfourier(fin2, fout2, 4);
          readBackTwoDFourier("cv2_fourier_fit.dat", nx, ny, x_min, dx, y_min, dy, false, der2_smoothed);
        }
        std::vector<double> bias_vals =
          integrateFromDerivatives(cv1_vals, cv2_vals,
            der1_smoothed, der2_smoothed);
        for (size_t k = 0; k < bias_vals.size(); ++k) {
          pointOfile_.printField("bias", -1.0 * bias_vals[k])
            .printField(getPntrToArgument(0)->getName(), cv1_vals[k])
            .printField(getPntrToArgument(1)->getName(), cv2_vals[k])
            .printField("der_" + getPntrToArgument(0)->getName(), -1.0 * der1_smoothed[k])
            .printField("der_" + getPntrToArgument(1)->getName(), -1.0 * der2_smoothed[k])
            .printField();
        }
        pointOfile_.flush();
        log.printf("Successfully wrote 2D POINT file (smoothed + optional 2D Fourier + integrated).\n");
        {
          const std::string cv1_name = getPntrToArgument(0)->getName();
          const std::string cv2_name = getPntrToArgument(1)->getName();
          OFile comb; comb.link(*this).open("DERIV_COMBINED_2D.dat");
          comb.fmtField(fmt);
          for (size_t k = 0; k < cv1_vals.size(); ++k) {
            comb.printField(cv1_name, cv1_vals[k])
              .printField(cv2_name, cv2_vals[k])
              .printField("der_" + cv1_name, der1_smoothed[k])
              .printField("der_" + cv2_name, der2_smoothed[k])
              .printField();
          }
          comb.flush();
          log.printf("Wrote merged derivatives to DERIV_COMBINED_2D.dat\n");
        }
        return;
      }
      error("runNeuralNetworkTrainingAndOutput currently only supports ncv == 1 or 2.");
    }
    void DEEPIBP::twodfourier(IFile& infile, OFile& outfile, int n_terms) {
      log.printf(" start Fourier\n");
      std::vector<double> x_data, y_data, z_data;
      double x, y, z;
      while (true) {
        bool ok = true;
        ok &= infile.scanField("phi", x);
        ok &= infile.scanField("psi", y);
        ok &= infile.scanField("value", z);
        if (!ok) break;
        infile.scanField();
        x_data.push_back(x);
        y_data.push_back(y);
        z_data.push_back(z);
      }
      if (x_data.empty()) {
        log.printf(" Empty input, exiting program.\n");
        return;
      }
      log.printf("Reading data points: %zu\n", x_data.size());
      std::map<double, std::vector<int>> y_groups;
      for (int i = 0; i < y_data.size(); i++) {
        y_groups[y_data[i]].push_back(i);
      }
      int N_grid = 200;
      std::vector<double> x_grid(N_grid);
      double step = (2 * kPi) / (N_grid - 1);
      for (int i = 0; i < N_grid; i++) {
        x_grid[i] = -kPi + i * step;
      }
      for (auto& kv : y_groups) {
        double current_y = kv.first;
        std::vector<double> x_group, z_group;
        for (int idx : kv.second) {
          x_group.push_back(x_data[idx]);
          z_group.push_back(z_data[idx]);
        }
        int N = x_group.size();
        if (N < n_terms * 2) {
          log.printf(" y=%.3f Insufficient data for %d-th order Fourier fit(only %d points)\n", current_y, n_terms, N);
          continue;
        }
        Eigen::MatrixXd A(N, 2 * n_terms);
        Eigen::VectorXd Y(N);
        for (int row = 0; row < N; row++) {
          Y(row) = z_group[row];
          for (int n = 1; n <= n_terms; n++) {
            A(row, 2 * (n - 1)) = std::cos(n * x_group[row]);
            A(row, 2 * (n - 1) + 1) = std::sin(n * x_group[row]);
          }
        }
        Eigen::VectorXd coeffs = (A.transpose() * A).ldlt().solve(A.transpose() * Y);
        for (int j = 0; j < N_grid; j++) {
          double val = 0.0;
          for (int n = 1; n <= n_terms; n++) {
            double a = coeffs(2 * (n - 1));
            double b = coeffs(2 * (n - 1) + 1);
            val += a * std::cos(n * x_grid[j]) + b * std::sin(n * x_grid[j]);
          }
          outfile.printField("phi", x_grid[j])
            .printField("psi", current_y)
            .printField("fitted_value", val)
            .printField();
        }
        log.printf(" Fourier complete \n");
      }
    }
    void DEEPIBP::fitOneDimFourier(IFile& infile, OFile& outfile, int n_terms,
      double xmin_raw, double xmax_raw, int N_grid) {
      log.printf("Initiating one-dimensional Fourier fit (n_terms = %d, without integration)\n", n_terms);
      std::vector<double> x_norm_data, y_data;
      double xr, y;
      while (true) {
        bool ok = true;
        ok &= infile.scanField("phi", xr);
        ok &= infile.scanField("value", y);
        if (!ok) break;
        infile.scanField();
        double L = std::max(1e-12, xmax_raw - xmin_raw);
        double mid = 0.5 * (xmax_raw + xmin_raw);
        double xn = (xr - mid) * (2.0 * kPi / L);
        xn = std::fmod(xn + kPi, 2.0 * kPi) - kPi;
        x_norm_data.push_back(xn);
        y_data.push_back(y);
      }
      if (x_norm_data.empty()) {
        log.printf(" Empty input, exiting program.\n");
        return;
      }
      log.printf(" Reading data points: %zu\n", x_norm_data.size());
      const int N = static_cast<int>(x_norm_data.size());
      Eigen::MatrixXd A(N, 2 * n_terms);
      Eigen::VectorXd Y(N);
      for (int i = 0; i < N; ++i) {
        Y(i) = y_data[i];
        for (int n = 1; n <= n_terms; ++n) {
          A(i, 2 * (n - 1)) = std::cos(n * x_norm_data[i]);
          A(i, 2 * (n - 1) + 1) = std::sin(n * x_norm_data[i]);
        }
      }
      Eigen::VectorXd coeffs = (A.transpose() * A).ldlt().solve(A.transpose() * Y);
      const double L = std::max(1e-12, xmax_raw - xmin_raw);
      const double mid = 0.5 * (xmax_raw + xmin_raw);
      for (int i = 0; i < N_grid; ++i) {
        double x_out = xmin_raw + i * (xmax_raw - xmin_raw) / (N_grid - 1);
        double xn = (x_out - mid) * (2.0 * kPi / L);
        xn = std::fmod(xn + kPi, 2.0 * kPi) - kPi;
        double val_fit = 0.0;
        for (int n = 1; n <= n_terms; ++n) {
          double a = coeffs(2 * (n - 1));
          double b = coeffs(2 * (n - 1) + 1);
          val_fit += a * std::cos(n * xn) + b * std::sin(n * xn);
        }
        outfile.printField("x", x_out)
          .printField("y_fit", val_fit)
          .printField();
      }
      log.printf("1D Fourier fitting completed, results have been output\n");
    }
    std::vector<double> DEEPIBP::integrateFromDerivatives(
      const std::vector<double>& cv1,
      const std::vector<double>& cv2,
      const std::vector<double>& der_cv1,
      const std::vector<double>& der_cv2)
    {
      if (cv1.size() != cv2.size() || cv1.size() != der_cv1.size() || cv1.size() != der_cv2.size()) {
        error("integrateFromDerivatives: Input vectors have inconsistent lengths");
      }
      std::vector<double> x_unique = cv1;
      std::vector<double> y_unique = cv2;
      std::sort(x_unique.begin(), x_unique.end());
      std::sort(y_unique.begin(), y_unique.end());
      x_unique.erase(std::unique(x_unique.begin(), x_unique.end()), x_unique.end());
      y_unique.erase(std::unique(y_unique.begin(), y_unique.end()), y_unique.end());
      int nx = x_unique.size();
      int ny = y_unique.size();
      int N = nx * ny;
      if (N != cv1.size()) {
        std::cerr << " Warning: The number of data points (" << cv1.size() << ") does not equal nx*ny (" << N << "). Please check the sorting logic.\n";
      }
      auto idx = [nx](int i, int j) { return i * nx + j; };
      std::vector<double> F_xy(N, 0.0);
      std::vector<double> F_yx(N, 0.0);
      for (int i = 0; i < ny; i++) {
        for (int j = 1; j < nx; j++) {
          double dx = x_unique[j] - x_unique[j - 1];
          F_xy[idx(i, j)] = F_xy[idx(i, j - 1)] + 0.5 * (der_cv1[idx(i, j)] + der_cv1[idx(i, j - 1)]) * dx;
        }
      }
      for (int j = 0; j < nx; j++) {
        for (int i = 1; i < ny; i++) {
          double dy = y_unique[i] - y_unique[i - 1];
          F_xy[idx(i, j)] = F_xy[idx(i - 1, j)] + 0.5 * (der_cv2[idx(i, j)] + der_cv2[idx(i - 1, j)]) * dy;
        }
      }
      for (int j = 0; j < nx; j++) {
        for (int i = 1; i < ny; i++) {
          double dy = y_unique[i] - y_unique[i - 1];
          F_yx[idx(i, j)] = F_yx[idx(i - 1, j)] + 0.5 * (der_cv2[idx(i, j)] + der_cv2[idx(i - 1, j)]) * dy;
        }
      }
      for (int i = 0; i < ny; i++) {
        for (int j = 1; j < nx; j++) {
          double dx = x_unique[j] - x_unique[j - 1];
          F_yx[idx(i, j)] = F_yx[idx(i, j - 1)] + 0.5 * (der_cv1[idx(i, j)] + der_cv1[idx(i, j - 1)]) * dx;
        }
      }
      std::vector<double> F_avg(N, 0.0);
      for (int k = 0; k < N; k++) {
        F_avg[k] = 0.5 * (F_xy[k] + F_yx[k]);
      }
      return F_avg;
    }
    void DEEPIBP::update()
    {
      if (!subnn && !guess)
      {
        unsigned ncv = getNumberOfArguments();
        bool nowAddAHill;
        if (getStep() % current_stride_ == 0 && populated_gaussian < num_sub && !isFirstStep_) {
          nowAddAHill = true;
        }
        else {
          nowAddAHill = false;
          isFirstStep_ = false;
        }
        std::vector<double> cv(ncv);
        for (unsigned i = 0; i < ncv; ++i) cv[i] = getArgument(i);
        if (nowAddAHill) {
          double height = height_sub;
          std::vector<double> thissigma = sigma_sub;
          bool multivariate = false;
          Gaussian newhill = Gaussian(multivariate, height, cv, thissigma);
          addGaussian(newhill);
          writeGaussian(newhill, hillsOfile_);
          populated_gaussian++;
          if (populated_gaussian == num_sub && !sumhills_called_) {
            log.printf("Reached target number of Gaussians (%u). Calling sumhills...\n", num_sub);
            sumhills();
            sumhills_called_ = true;
          }
        }
      }
    }
  }
}
