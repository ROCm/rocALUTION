#ifndef PARALUTION_ITER_CTRL_HPP_
#define PARALUTION_ITER_CTRL_HPP_

#include <vector>
#include <string>

namespace paralution {

/// Iteration control for iterative solvers, monitor the 
/// residual (L2 norm) behavior
class IterationControl {

public:

  IterationControl();
  ~IterationControl();

  /// Initialize with absolute/relative/divergence 
  /// tolerance and maximum number of iterations
  void Init(const double abs,
            const double rel,
            const double div,
            const int max);

  /// Initialize with absolute/relative/divergence 
  /// tolerance and minimum/maximum number of iterations
  void Init(const double abs,
            const double rel,
            const double div,
            const int min,
            const int max);

  /// Initialize with absolute/relative/divergence tolerance
  void InitTolerance(const double abs,
                     const double rel,
                     const double div);

  /// Set the minimum number of iterations
  void InitMinimumIterations(const int min);

  /// Set the maximal number of iterations
  void InitMaximumIterations(const int max);

  /// Get the minimum number of iterations
  int GetMinimumIterations(void) const;

  /// Get the maximal number of iterations
  int GetMaximumIterations(void) const;

  /// Initialize the initial residual
  bool InitResidual(const double res);

  /// Clear (reset)
  void Clear(void);

  /// Check the residual (this count also the number of iterations)
  bool CheckResidual(const double res);

  /// Check the residual and index value for the inf norm 
  /// (this count also the number of iterations)
  bool CheckResidual(const double res, const int index);

  /// Check the residual (without counting the number of iterations)
  bool CheckResidualNoCount(const double res) const;

  /// Record the history of the residual
  void RecordHistory(void);

  /// Write the history of the residual to an ASCII file
  void WriteHistoryToFile(const std::string filename) const;

   /// Provide verbose output of the solver (iter, residual)
  void Verbose(const int verb=1);

  /// Print the initialized information of the iteration control
  void PrintInit(void);

  /// Print the current status (is the any criteria reached or not)
  void PrintStatus(void);

  // Return the iteration count
  int GetIterationCount(void);

  // Return the current residual
  double GetCurrentResidual(void);

  // Return the current status
  int GetSolverStatus(void);

  /// Return absolute maximum index of residual vector when using Linf norm
  int GetAmaxResidualIndex(void);

private:

  /// Verbose flag
  /// verb == 0 no output
  /// verb == 1 print info about the solver (start,end);
  /// verb == 2 print (iter, residual) via iteration control;
  int verb_;

  /// Iteration count
  int iteration_;

  /// Flag == true after calling InitResidual()
  bool init_res_;

  /// Initial residual
  double initial_residual_;

  /// Absolute tolerance
  double absolute_tol_;
  /// Relative tolerance
  double relative_tol_;
  /// Divergence tolerance
  double divergence_tol_;
  /// Minimum number of iteration
  int minimum_iter_;
  /// Maximum number of iteration
  int maximum_iter_;

  /// Indicator for the reached criteria:
  /// 0 - not yet;
  /// 1 - abs tol is reached;
  /// 2 - rel tol is reached;
  /// 3 - div tol is reached;
  /// 4 - max iter is reached
  int reached_;

  /// STL vector keeping the residual history
  std::vector<double> residual_history_;

  /// Current residual (obtained via CheckResidual())
  double current_res_;

  /// Current residual, index for the inf norm (obtained via CheckResidual())
  int current_index_;

  /// Flag == true then the residual is recorded in the residual_history_ vector
  bool rec_;

};


}

#endif // PARALUTION_ITER_CTRL_HPP_
