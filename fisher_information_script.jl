import QuantumCollocation as QC
import NamedTrajectories as NT
import TrajectoryIndexingUtils as NTidx
import JLD2

include("utils.jl")
include("system.jl")
include("objectives.jl")
include("constraints.jl")

V = 10.
p_max = 5
system = ShakenLatticeSystem1D(V, p_max; acc=0.0, include_acc_derivative=true, sparse=false)
mid = system.params[:mid]
dim = system.params[:dim]

#Z_guess = NT.load_traj("./interferometer/176-5.0_352-5.0.jld2")
Z_guess = NT.load_traj("./interferometer/save.jld2")
dts = vec(Z_guess.dts)

duration = sum(dts) # in units of 1/E_R
T = length(dts)
dt = duration / (T-1)
dts = zeros(T) .+ dt
dt_bound = (dt, dt)
times = cumsum(dts) - dts

psi0 = get_bloch_state(system; lvl=0)
if system.params[:accelerated]
    append!(psi0, zeros(dim))
end

time_flight = 2pi * 5.

jumps = [(176, time_flight), (352, time_flight)]
cuts = [jump[1] for jump in jumps]
full_times = get_times(dts, jumps)
G = get_shaken_lattice_propagator(system, times, jumps, 10000)

MZFI = (8pi*(full_times[end]/2)^2)^2

function Fisher(psi, dpsi)
    eps = 0.0
    P = abs2.(psi)
    D = 2*real.(conj.(psi) .* dpsi)
    F = (1 ./ (P .+ eps))' * D.^2
    return F / MZFI
end 

fisher_loss = psi_dpsi -> -Fisher(psi_dpsi[1:dim], psi_dpsi[dim+1:2*dim])

J = NameComponentObjective(
    :psi_iso,
    [1:2*dim...],
    [T],
    fisher_loss;
    is_iso=true
)
J += QC.QuadraticRegularizer(:dda, Z_guess, 1e-8/T)

integrators = [
    QC.QuantumStatePadeIntegrator(
        system,
        :psi_iso,
        (:a, :acc),
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

dynamics = QC.QuantumDynamics(
    integrators,
    Z_guess;
    cuts=cuts
)

constraints = [
    OmegaAbsConstraint(1.0, Z_guess, Z_guess.components[:a]),
    vcat([get_link_constraints(
        :psi_iso, 
        Z_guess, 
        c, 
        g, 
        (; a=[1.0,0.0]), 
        (; a=[1.0,0.0]); 
        hard_equality_constraint=true)
        for (c, g) in zip(cuts, G)]...)...,
    TimeAffineLinearControlConstraint(:acc, 1, Z_guess; jumps=jumps)
]

options = QC.Options(
    max_iter=10000,
)

prob = QC.QuantumControlProblem(
    system, 
    Z_guess, 
    J, 
    dynamics;
    constraints=constraints,
    options=options,
)

QC.solve!(prob)

JLD2.save("./interferometer/save.jld2", prob.trajectory)