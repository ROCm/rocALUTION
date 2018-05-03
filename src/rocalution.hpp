#ifndef ROCALUTION_ROCALUTION_HPP_
#define ROCALUTION_ROCALUTION_HPP_

#include "base/version.hpp"
#include "base/backend_manager.hpp"
#include "base/parallel_manager.hpp"

#include "base/operator.hpp"
#include "base/vector.hpp"

#include "base/matrix_formats.hpp"
#include "base/global_matrix.hpp"
#include "base/local_matrix.hpp"

#include "base/global_vector.hpp"
#include "base/local_vector.hpp"

#include "base/local_stencil.hpp"
#include "base/stencil_types.hpp"

#include "solvers/solver.hpp"
#include "solvers/iter_ctrl.hpp"
#include "solvers/chebyshev.hpp"
#include "solvers/mixed_precision.hpp"
#include "solvers/krylov/cg.hpp"
#include "solvers/krylov/fcg.hpp"
#include "solvers/krylov/cr.hpp"
#include "solvers/krylov/bicgstab.hpp"
#include "solvers/krylov/bicgstabl.hpp"
#include "solvers/krylov/qmrcgstab.hpp"
#include "solvers/krylov/gmres.hpp"
#include "solvers/krylov/fgmres.hpp"
#include "solvers/krylov/idr.hpp"
#include "solvers/multigrid/base_multigrid.hpp"
#include "solvers/multigrid/base_amg.hpp"
#include "solvers/multigrid/multigrid.hpp"
#include "solvers/multigrid/amg.hpp"
#include "solvers/multigrid/ruge_stueben_amg.hpp"
#include "solvers/multigrid/pairwise_amg.hpp"
#include "solvers/multigrid/global_pairwise_amg.hpp"
#include "solvers/direct/inversion.hpp"
#include "solvers/direct/lu.hpp"
#include "solvers/direct/qr.hpp"

#include "solvers/preconditioners/preconditioner.hpp"
#include "solvers/preconditioners/preconditioner_blockjacobi.hpp"
#include "solvers/preconditioners/preconditioner_ai.hpp"
#include "solvers/preconditioners/preconditioner_as.hpp"
#include "solvers/preconditioners/preconditioner_multicolored.hpp"
#include "solvers/preconditioners/preconditioner_multicolored_gs.hpp"
#include "solvers/preconditioners/preconditioner_multicolored_ilu.hpp"
#include "solvers/preconditioners/preconditioner_multielimination.hpp"
#include "solvers/preconditioners/preconditioner_saddlepoint.hpp"
#include "solvers/preconditioners/preconditioner_blockprecond.hpp"

#include "utils/allocate_free.hpp"
#include "utils/time_functions.hpp"
#include "utils/types.hpp"


#endif // ROCALUTION_ROCALUTION_HPP_
