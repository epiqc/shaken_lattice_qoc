import QuantumCollocation as QC
include("system.jl")


function population_loss(pops)
    return x -> sum(abs2.(abs2.(x) .- pops))
end


function split_protocol(
    system::QC.QuantumSystem,
    dts,
    psi0=get_bloch_groundstate(system);
    dt_bound=(minimum(dts), maximum(dts))
    a=vcat(ones(length(dts))', zeros(length(dts))'),
    da=NT.derivative(a, dts),
    dda=NT.derivative(da, dts),
    dda_bound=fill(100., 2),
    pade_order=4
)
    mid = system.params[:mid]
    psi0_iso = QC.ket_to_iso(psi0)
    psi_iso = QC.rollout(
        psi0_iso, a, dts, system; 
        integrator=G -> QC.nth_order_pade(G, pade_order)
    )

    comps = (
        psi_iso = psi_iso,
        a = a,
        da = da,
        dda = dda,
        dts = dts
    )
    initial = (
        psi_iso = psi0_iso,
        a = [1.; 0.],
        da = zeros(2)
    )
    final = (;
        a = [1.; 0.],
        da = zeros(2)
    )
    bounds = (
        a = a_bound,
        dda = dda_bound,
        dts = dt_bound
    )
    Z_guess = NT.NamedTrajectory(
        comps;
        controls=(:dda),
        timestep=:dts,
        bounds=bounds,
        initial=initial,
        final=final,
        goal=(;)
    )

    J = NameComponentObjective(
        :psi_iso, 
        [mid-1, mid, mid+1], 
        population_loss([0.5]), 
        1.0; 
        is_iso=true
    )

    integrators = [
        QC.QuantumStatePadeIntegrator(
            system,
            :psi_iso,
            :a,
            :dts;
            order=4
        ),
        QC.DerivativeIntegrator(
            :a,
            :da,
            :dts,
            Z_guess
        ),
        QC.DerivativeIntegrator(
            :da,
            :dda,
            :dts,
            Z_guess
        )
    ]

    constraints = [
        OmegaAbsConstraint(1.0, Z_guess),
    ]

    # Ipopt options
    options = QC.Options(
        max_iter=600,
    )

    prob = QC.QuantumControlProblem(
        system, 
        Z_guess, 
        J, 
        integrators;
        constraints=constraints,
        options=options,
    )

    QC.solve!(prob)

    return prob.trajectory
end