import Pkg
Pkg.activate("../../")

import QuantumCollocation as QC
import NamedTrajectories as NT
import TrajectoryIndexingUtils as NTidx
import LinearAlgebra as LA
import JLD2


include("../../utils.jl")
include("../../system.jl")
include("../../objectives.jl")
include("../../constraints.jl")


function main()
    Z = NT.load_traj("101.jld2")
    T = Z.T
    T_step = 2
    save_every = 5
    T_end = 501

    phi_bound = [1.0 * pi]
    
    V = 10.
    trunc = 11
    # E_R [kHz] found in Weidner thesis
    system = ShakenLatticeSystem1D(
        V, 
        trunc; 
        acc=0.0, 
        bloch_basis=true,
        bloch_transformation_trunc=2*trunc,
        include_acc_derivative=true, 
        sparse=false)
    # middle index of statevector where p = 0
    mid = system.params[:mid]
    dim = system.params[:dim]
    B = system.params[:bloch_states][:,1:trunc]
    if system.params[:accelerated]
        B2 = blockdiagonal(B, B)
    end

    function Fisher(psi, dpsi)
        psi = B*psi
        dpsi = B*dpsi
        eps = 0.0
        P = abs2.(psi)
        D = 2*real.(conj.(psi) .* dpsi)
        F = (1 ./ (P .+ eps))' * D.^2
        return F
    end 

    integrators = nothing
    integrators = [
        QC.QuantumStatePadeIntegrator(
            system,
            :psi_iso,
            (:a, :acc),
            :dts;
            order=4
        )
    ]

    # Ipopt options
    options = QC.Options(
        max_iter=200,
    )

    i = 1
    while T <= T_end
        T = T + T_step
        duration = 2pi*(T-1)/1000
        Z_guess = trajectory_shrink_extend(Z, T)

        MZFI = (8pi*(duration/2)^2)^2
        fisher_loss = psi_dpsi -> -Fisher(psi_dpsi[1:dim], psi_dpsi[dim+1:2*dim]) / MZFI
        J1 = NameComponentObjective(
            :psi_iso,
            [1:2*dim...],
            [T],
            fisher_loss;
            is_iso=true
        )
        # convolution
        kernel = sinc_kernel(50., vec(Z_guess.dts))
        convolver = LA.I(Z_guess.T) - kernel
        convolver = convolver' * convolver
        J2 = QuadraticObjective(:phi, Z_guess, convolver, [1], 1:Z_guess.T; Q=1.0/T)
        J = J1 + J2

        dynamics = QC.QuantumDynamics(
            integrators,
            Z_guess
        )

        constraints = [
            IQPhiConstraint(:a, :phi, Z_guess),
            TimeAffineLinearControlConstraint(:acc, 1, Z_guess),
            custom_bounds_constraint(:phi, Z_guess, Int64[], phi_bound)
        ]

        # defining quantum control problem
        prob = nothing
        prob = QC.QuantumControlProblem(
            system, 
            Z_guess, 
            J, 
            dynamics;
            constraints=constraints,
            options=options,
        )

        QC.solve!(prob)

        Z = prob.trajectory
        if i % save_every == 0
            s = """
            Duration (in ωᵣ):
            $duration
            MZFI (+-4k_L split):
            $MZFI
            Final CFI (normalized to MZFI):
            $(-J1.L(Z.datavec, Z))
            """
            write("$T.txt", s)

            JLD2.save("$T.jld2", Z)
        end
        i += 1
        println("\n\n")
    end
end


main()