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
    Z_guess = NT.load_traj("0.001.jld2")
    T = Z_guess.T
    dts = vec(Z_guess.dts)
    times = cumsum(dts) - dts
    save_every = 1
    flight_times_start = 2pi*0.01
    flight_times_step = 2pi*0.01
    flight_times_end = 2pi*0.2
    flight_times = collect(flight_times_start:flight_times_step:flight_times_end)

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
        max_iter=400,
    )

   for (i, flight_time) in enumerate(flight_times)
        T_flight = Int(round(flight_time/2pi*2000; digits=0))
        jumps = [(div(T, 2), flight_time)]
        cuts = [jump[1] for jump in jumps]
        full_times = get_times(dts, jumps)
        G = get_shaken_lattice_propagator(system, times, jumps, T_flight)

        MZFI = (8pi*(full_times[end]/2)^2)^2
        fisher_loss = psi_dpsi -> -Fisher(psi_dpsi[1:dim], psi_dpsi[dim+1:2*dim]) / MZFI
        log_sensitivity_loss = psi_dpsi -> -0.5 * log10(Fisher(psi_dpsi[1:dim], psi_dpsi[dim+1:2*dim])) + 0.5 * log10(MZFI)
        J1 = NameComponentObjective(
            :psi_iso,
            [1:2*dim...],
            [T],
            log_sensitivity_loss;
            is_iso=true
        )
        # convolution
        kernel = sinc_kernel(50., dts)
        convolver = LA.I(T) - kernel
        convolver = convolver' * convolver
        J2 = QuadraticObjective(:phi, Z_guess, convolver, [1], 1:T; Q=50.0/T)
        J = J1 + J2

        dynamics = QC.QuantumDynamics(
            integrators,
            Z_guess;
            cuts=cuts
        )

        constraints = [
            IQPhiConstraint(:a, :phi, Z_guess),
            vcat([get_link_constraints(
                :psi_iso, 
                Z_guess, 
                c, 
                g, 
                (; phi=[0.]), 
                (; phi=[0.]); 
                hard_equality_constraint=true)
                for (c, g) in zip(cuts, G)]...)...,
            TimeAffineLinearControlConstraint(:acc, 1, Z_guess; jumps=jumps),
            custom_bounds_constraint(:phi, Z_guess, vcat(cuts, cuts .+ 1), phi_bound)
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
            Flight time (in ωᵣ/2pi):
            $(flight_time/2pi)
            MZFI (+-4k_L split):
            $MZFI
            Final CFI (normalized to MZFI):
            $(-fisher_loss(QC.iso_to_ket(Z.psi_iso[:,end])))
            Final log10(δa):
            $(J1.L(Z.datavec, Z))
            """
            fname = "$(round(flight_time/2pi, digits=2))"
            write("$fname.txt", s)

            JLD2.save("$fname.jld2", Z)
        end
        Z_guess = Z
        GC.gc()
        println("\n\n")
    end
end


main()