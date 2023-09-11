import Pkg
Pkg.activate("../../")

import QuantumCollocation as QC
import NamedTrajectories as NT
import LinearAlgebra as LA
import JLD2


include("../../utils.jl")
include("../../system.jl")
include("../../constraints.jl")
include("../../objectives.jl")


function main()
    T_step = 10
    Z_guess = NT.load_traj("0.4wr.jld2")

    V = 10.
    trunc = 11
    system = ShakenLatticeSystem1D(V, trunc; bloch_basis=true, bloch_transformation_trunc=2*trunc)
    mid = system.params[:mid]
    dim = system.params[:dim]

    # 1 - subspace projection
    R = 1.0 * collect(LA.I(dim))
    R[8,8] = R[9,9] = 0.
    R = QC.QuantumSystems.iso(R)

    integrators = [
        QC.QuantumStatePadeIntegrator(
            system,
            :psi_iso,
            :a,
            :dts;
            order=4
        )
    ]

    # Ipopt options
    options = QC.Options(
        max_iter=200,
    )

    for i=1:5
        # shrink trajectory
        Z_guess = trajectory_shrink_extend(Z_guess, Z_guess.T - T_step)
        println("T=$(Z_guess.T)")

        # subspace penalty
        J1 = QuadraticObjective(:psi_iso, Z_guess, R; Q=200.0)

        # convolution penalty
        kernel = sinc_kernel(40., vec(Z_guess.dts))
        convolver = LA.I(Z_guess.T) - kernel
        convolver = convolver' * convolver
        J2 = QuadraticObjective(:phi, Z_guess, convolver, [1], 1:Z_guess.T; Q=50.0/Z_guess.T)

        # objective
        J = J1 + J2

        # constraints
        constraints = [
            IQPhiConstraint(:a, :phi, Z_guess),
            FinalYZGreatCircleConstraint(:psi_iso, [8, 9], Z_guess)
        ]

        # defining quantum control problem
        prob = nothing
        GC.gc()
        prob = QC.QuantumControlProblem(
            system, 
            Z_guess, 
            J, 
            integrators;
            constraints=constraints,
            options=options,
        )

        QC.solve!(prob)

        Z = prob.trajectory
        duration = sum(Z.dts) - Z.dts[end]
        duration_wr = round(duration/2pi; digits=2)
        s = """
        Final subspace infidelity (YZ greatcircle) in %:
        $(J1.L(Z.datavec, Z))
        """
        write("$(duration_wr)wr.txt", s)

        JLD2.save("$(duration_wr)wr.jld2", Z)

        Z_guess = Z
        println("\n\n")
    end
end


main()