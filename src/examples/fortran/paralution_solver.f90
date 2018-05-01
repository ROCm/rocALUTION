program rocalution_solver

  use, intrinsic :: ISO_C_BINDING, only : C_INT, C_PTR, C_DOUBLE, C_CHAR, C_NULL_CHAR, C_LOC

  implicit none

  interface
    subroutine rocalution_init() BIND(C)
    end subroutine rocalution_init

    subroutine rocalution_stop() BIND(C)
    end subroutine rocalution_stop

    subroutine rocalution_fortran_solve_coo( n, m, nnz, solver, mformat, preconditioner, pformat,    &
    &                                        rows, cols, rval, rhs, atol, rtol, div, maxiter, basis, &
    &                                        p, q, x, iter, resnorm, ierr ) BIND(C)

      use, intrinsic :: ISO_C_BINDING, only : C_INT, C_PTR, C_DOUBLE, C_CHAR

      integer(kind=C_INT), value, intent(in)  :: n, m, nnz, maxiter, basis, p, q
      real(kind=C_DOUBLE), value, intent(in)  :: atol, rtol, div
      integer(kind=C_INT),        intent(out) :: iter, ierr
      real(kind=C_DOUBLE),        intent(out) :: resnorm
      type(C_PTR),         value, intent(in)  :: rows, cols, rval, rhs
      type(C_PTR),         value              :: x
      character(kind=C_CHAR)                  :: solver, mformat, preconditioner, pformat

    end subroutine rocalution_fortran_solve_coo
  end interface

  integer, parameter    :: infile = 10
  integer(kind=C_INT)   :: n, m, nnz, fnz, i, j, iter, ierr
  real(kind=C_DOUBLE)   :: resnorm
  integer, dimension(8) :: tbegin, tend
  real(kind=8)          :: tsolver

  logical               :: sym = .false.

  integer(kind=C_INT), allocatable, target :: rows(:), cols(:)
  real(kind=C_DOUBLE), allocatable, target :: ival(:), rval(:), cval(:)
  real(kind=C_DOUBLE), allocatable, target :: rhs(:), x(:)

  character(len=10)  :: rep
  character(len=7)   :: field
  character(len=19)  :: symm
  character(len=128) :: arg


  ! Get command line option to read file
  do i = 1, iargc()
    call getarg(i, arg)
  end do
  open( unit = infile, file = arg, action = 'read', status = 'old' )

  call mminfo( infile, rep, field, symm, n, m, fnz )

  nnz = fnz
  if ( symm .eq. 'symmetric' .or. symm .eq. 'hermitian' ) then
    nnz = 2 * ( fnz - n ) + n
    sym = .true.
  end if

  ! Allocate memory for COO format specific arrays
  allocate( rows(nnz), cols(nnz) )
  allocate( ival(nnz), rval(nnz), cval(nnz) )

  ! Read COO matrix in matrix market format
  call mmread( infile, rep, field, symm, n, m, fnz, nnz, &
  &            rows, cols, ival, rval, cval )

  ! Close file
  close( unit = infile )

  ! Deallocate arrays we do not use
  deallocate( ival, cval )

  ! Fill 2nd half of matrix if symmetric
  if ( sym ) then
    j = fnz + 1
    do i = 1, fnz
      if ( rows(i) .ne. cols(i) ) then
        rows(j) = cols(i)
        cols(j) = rows(i)
        rval(j) = rval(i)
        j = j + 1
      end if
    end do
  end if

  ! Allocate and initialize rhs and solution vector
  allocate( rhs(n), x(n) )
  do i = 1, n
    rhs(i) = 1._C_DOUBLE
    x(i)   = 0._C_DOUBLE
  end do

  ! Print L2 norm of solution vector
  write(*,fmt='(A,F0.2)') '(Fortran) Initial L2 Norm(x) = ', sqrt( sum( x**2 ) )

  ! Initialize rocALUTION backend
  call rocalution_init

  call date_and_time(values = tbegin)

  ! Run rocalution C function for COO matrices
  ! Doing a GMRES with MultiColored ILU(1,2) preconditioner
  ! Check rocalution documentation for a detailed argument explanation
  call rocalution_fortran_solve_coo( n, m, nnz,                                          &
  &                                  'CG' // C_NULL_CHAR,                                &
  &                                  'CSR' // C_NULL_CHAR,                               &
  &                                  'MultiColoredILU' // C_NULL_CHAR,                   &
  &                                  'CSR' // C_NULL_CHAR,                               &
  &                                  C_LOC(rows), C_LOC(cols), C_LOC(rval), C_LOC(rhs),  &
  &                                  1e-15_C_DOUBLE, 1e-8_C_DOUBLE, 1e+8_C_DOUBLE, 5000, &
  &                                  30, 0, 1, C_LOC(x), iter, resnorm, ierr )

  call date_and_time(values = tend)

  tbegin = tend - tbegin
  tsolver = 0.001 * tbegin(8) + tbegin(7) + 60 * tbegin(6) + 3600 * tbegin(5)
  write(*,fmt='(A,F0.2,A)') '(Fortran) Solver ended after ', tsolver,'sec.'

  ! Print solver details
  if ( ierr .eq. 0 ) then
    write(*,fmt='(A,I0,A,E11.5,A)') '(Fortran) Solver took ', iter, ' iterations with residual norm ', resnorm, '.'
    write(*,fmt='(A,F0.2)') '(Fortran) Final L2 Norm(x)   = ', sqrt( sum( x**2 ) )
  else
    write(*,fmt='(A,I0)') '(Fortran) Solver returned status code ', ierr
  end if

  do i=1,n
    write(10,*) x(i)
  end do

  deallocate( rows, cols, rval, rhs, x )

  ! Stop rocALUTION backend
  call rocalution_stop

end program rocalution_solver

