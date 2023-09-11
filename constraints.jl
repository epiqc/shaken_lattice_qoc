import QuantumCollocation as QC
import NamedTrajectories as NT
import TrajectoryIndexingUtils as NTidx
import Ipopt
import MathOptInterface as MOI
import ForwardDiff as FD


include("utils.jl")


function OmegaAbsConstraint(
    R,
    traj,
    comps=traj.components[:a],
    zdim=traj.dim,
    T=traj.T,
    times=1:T
)
    ineq_con = QC.ComplexModulusContraint(
        R=R,
        comps=comps,
        times=times,
        zdim=zdim,
        T=T
    )
    props = [getproperty(ineq_con, p) for p in propertynames(ineq_con)]
    eq_con = QC.NonlinearEqualityConstraint(props...)
    return eq_con
end


function PhiFunctionBoundConstraint(
    f::Function,
    traj::NT.NamedTrajectory;
    comp=traj.components[:a][1], # need component of I in z_t
    time_comp=traj.components[:dts][1],
    zdim=traj.dim,
    T=traj.T,
    times=1:T
)
    params = Dict{Symbol, Any}()
    params[:type] = :PhaseFunctionBoundConstraint
    params[:comp] = comp
    params[:time_comp] = time_comp
    params[:times] = times
    params[:zdim] = zdim
    params[:T] = T

    eps = 1e-8
    gₜ(t, I) = f(t) - acos(clamp(I, -1., 1.))
    ∂gₜ(I) = 1/sqrt(1. - I^2 + eps)
    μₜ∂²gₜ(μ, I) = μ*I/(sqrt(1-I^2+eps)^3)

    time_comps = [NTidx.index(t, time_comp, zdim) for t in times]

    @views function g(Z⃗)
        r = zeros(length(times))
        true_times = cumsum(Z⃗[time_comps]) .- Z⃗[time_comp]
        for (i, t) ∈ enumerate(times)
            I = Z⃗[NTidx.index(t, comp, zdim)]
            true_time = true_times[t]
            r[i] = gₜ(true_time, I)[1]
        end
        return r
    end

    ∂g_structure = [(i, NTidx.index(t, comp, zdim)) for (i,t) in enumerate(times)]
    @views function ∂g(Z⃗; ipopt=true)
        ∂ = SA.spzeros(length(times), zdim * T)
        for (i, t) ∈ enumerate(times)
            I_comp = NTidx.index(t, comp, zdim)
            I = Z⃗[I_comp]
            ∂[i, I_comp] = ∂gₜ(I)
        end
        if ipopt
            return [∂[i, j] for (i, j) in ∂g_structure]
        else
            return ∂
        end
    end

    μ∂²g_structure = [(NTidx.index(t, comp, zdim), NTidx.index(t, comp, zdim)) for t in times]
    @views function μ∂²g(Z⃗, μ; ipopt=true)
        μ∂² = SA.spzeros(zdim * T, zdim * T)
        for (i, t) ∈ enumerate(times)
            I_comp = NTidx.index(t, comp, zdim)
            I = Z⃗[I_comp]
            μ∂²[I_comp, I_comp] += μₜ∂²gₜ(μ[i], I)
        end
        if ipopt
            return [μ∂²[i, j] for (i, j) in μ∂²g_structure]
        else
            return μ∂²
        end
    end

    return QC.NonlinearInequalityConstraint(
        g,
        ∂g,
        ∂g_structure,
        μ∂²g,
        μ∂²g_structure,
        length(times),
        params
    )
end


function NameComponentPathConstraint(
    name::Symbol,
    comps::AbstractVector{Int},
    traj::NT.NamedTrajectory,
    f::AbstractVector, # need to be functions describing paths
    h::AbstractVector; # need to be functions describing mapping of components
    zdim=traj.dim,
    T=traj.T,
    ts=1:T,
    is_iso=false
)
    params = Dict(
        :type => :NameComponentPathConstraint,
        :name => name,
        :comps => comps,
        :is_iso => is_iso
    )

    name_comps = traj.components[name]
    dt_comp = traj.components[traj.timestep][1]
    dts_slice = [NTidx.index(t, dt_comp, zdim) for t=1:maximum(ts)]
    num_paths = length(f)
    @assert num_paths == length(comps) == length(h)

    if is_iso
        h_til = [x -> h[i](QC.iso_to_ket(x)[1]) for i=1:num_paths]
        dh = [x -> FD.gradient(h_til[i], x) for i=1:num_paths]
        ddh = [x -> FD.hessian(h_til[i], x) for i=1:num_paths]
    else
        h_til = h
        dh = [x -> FD.derivative(h[i], x) for i=1:num_paths]
        ddh = [x -> FD.derivative(dh[i], x) for i=1:num_paths]
    end
    df = [time -> FD.derivative(f[i], time) for i=1:num_paths]
    ddf = [time -> FD.derivative(df[i], time) for i=1:num_paths]

    μₜ∂²gₜ(μ, x, t) = μ * [ddh(x), -ddf(t)]

    function get_comp_idc_traj(name_slice, comp_idx)
        name_dim = length(name_slice)
        comp = comps[comp_idx]
        if is_iso
            return name_slice[[comp, comp + div(name_dim, 2)]]
        else
            return name_slice[comp]
        end
    end

    get_con_idx(i_t, i) = (i_t-1) * num_paths + i
    num_cons = length(ts) * num_paths

    @views function g(Z⃗::AbstractVector{<:Real})
        r = zeros(num_cons)
        dts = Z⃗[dts_slice]
        con_idx = 0
        for (i_t, t) in enumerate(ts)
            time = sum(dts[1:t]) - dts[t]
            name_slice = NTidx.slice(t, name_comps, zdim)
            for i=1:num_paths
                x = Z⃗[get_comp_idc_traj(name_slice, i)]
                con_idx += 1
                r[con_idx] = h_til[i](x) - f[i](time)
            end
        end
        return r
    end
  
    ∂g_structure = []
    μ∂²g_structure = []
    con_idx = 0
    for (i_t, t) in enumerate(ts) 
        name_slice = NTidx.slice(t, name_comps, zdim)
        for i=1:num_paths
            con_idx += 1
            comp_idc = get_comp_idc_traj(name_slice, i)
            # relevant loop in case that is_iso
            for (i_comp, comp_idx) in enumerate([comp_idc...])
                push!(∂g_structure, (con_idx, comp_idx))
                append!(μ∂²g_structure, [(comp_idc[j], comp_idc) for j=i:length(comp_idc)])
            end
            for dt_idx in dts_slice[1:t-1]
                push!(∂g_structure, (con_idx, dt_idx))
            end
        end
    end
    append!(μ∂²g_structure, [(dts_slice[i], dts_slice[j]) for i=1:maximum(ts) for j=1:i])

    @views function ∂g(Z⃗; ipopt=true)
        ∂ = SA.spzeros(num_cons, zdim * T)
        dts = Z⃗[dts_slice]
        con_idx = 0
        for (i_t, t) in enumerate(ts)
            time = sum(dts[1:t]) - dts[t]
            name_slice = NTidx.slice(t, name_comps, zdim)
            for i=1:num_paths
                comp_idc = get_comp_idc_traj(name_slice, i)
                x = Z⃗[comp_idc]
                con_idx += 1
                ∂[con_idx, comp_idc] = dh[i](x)
                ∂[con_idx, dts_slice[1:t-1]] .= -df[i](time)
            end
        end
        if ipopt
            return [∂[i, j] for (i, j) in ∂g_structure]
        else
            return ∂
        end
        return ∂
    end

    @views function μ∂²g(Z⃗, μ; ipopt=true)
        μ∂² = SA.spzeros(zdim * T, zdim * T)
        dts = Z⃗[dts_slice]
        con_idx = 0
        for (i_t, t) in enumerate(ts)
            time = sum(dts[1:t]) - dts[t]
            name_slice = NTidx.slice(t, name_comps, zdim)
            for i=1:num_paths
                comp_idc = get_comp_idc_traj(name_slice, i)
                x = Z⃗[comp_idc]
                con_idx += 1
                μ∂²[comp_idc, comp_idc] += μ[con_idx] * ddh[i](x)
                μ∂²[dts_slice[1:t-1], dts_slice[1:t-1]] .+= -μ[con_idx] * ddf[i](time)
            end
        end
        if ipopt
            return [μ∂²[i, j] for (i, j) in μ∂²g_structure]
        else
            return μ∂²
        end
    end

    return QC.NonlinearEqualityConstraint(
        g,
        ∂g,
        ∂g_structure,
        μ∂²g,
        μ∂²g_structure,
        num_cons,
        params
    )
end


struct TimeSymmetricControlConstraint <: QC.LinearConstraint
    equal_indices_pairs::AbstractVector{Tuple{Int, Int}}
    label::String

    function TimeSymmetricControlConstraint(
        name::Symbol,
        traj::NT.NamedTrajectory,
        label::String="time-symmetric control constraint"
    )
        comps = traj.components[name]
        equal_indices_pairs = [
            (NTidx.index(t, comps[j], traj.dim), NTidx.index(traj.T-t+1, comps[j], traj.dim))
             for j=1:traj.dims[name] for t=1:div(traj.T, 2)
            ]
        return new(equal_indices_pairs, label)
    end
end

function (con::TimeSymmetricControlConstraint)(
    opt::Ipopt.Optimizer,
    vars::Vector{MOI.VariableIndex},
    traj::NT.NamedTrajectory    
)
    for idc_pair in con.equal_indices_pairs
        a1 = MOI.ScalarAffineTerm(1.0, vars[idc_pair[1]])
        minusa2 = MOI.ScalarAffineTerm(-1.0, vars[idc_pair[2]])
        MOI.add_constraints(
            opt,
            MOI.ScalarAffineFunction([a1, minusa2], 0.0),
            MOI.EqualTo(0.0)
        )
    end
end


struct TimeAffineLinearControlConstraint <: QC.LinearConstraint
    comp_idc::Vector{Int}
    dt_idc::Vector{Int}
    a::Float64
    b::Float64
    ts::AbstractVector{Int}
    jumps::AbstractVector{Tuple{Int, Float64}}
    label::String

    function TimeAffineLinearControlConstraint(
        name::Symbol,
        comp::Int,
        traj::NT.NamedTrajectory,
        a=1.0,
        b=0.0;
        ts=1:traj.T,
        jumps=AbstractVector{Tuple{Int, Float64}}[],
        label::String="time affine linear control constraint"
    )
        @assert all([jump[1] in ts for jump in jumps])
        comp_idc = [NTidx.index(t, traj.components[name][comp], traj.dim) for t in ts]
        dt_idc = [NTidx.index(t, traj.components[traj.timestep][1], traj.dim) for t in ts[1:end-1]]
        return new(comp_idc, dt_idc, a, b, ts, jumps, label)
    end
end

function (con::TimeAffineLinearControlConstraint)(
    opt::Ipopt.Optimizer,
    vars::Vector{MOI.VariableIndex},
    traj::NT.NamedTrajectory
)
    minus_dts = [MOI.ScalarAffineTerm(-con.a, vars[dt_idx]) for dt_idx in con.dt_idc]
    b_accum = con.b

    a = MOI.ScalarAffineTerm(1.0, vars[con.comp_idc[1]])
    MOI.add_constraints(
        opt,
        MOI.ScalarAffineFunction([a], -b_accum),
        MOI.EqualTo(0.0)
    )

    minus_dts_idc = Int[]
    for (t_idx, comp_idx) in enumerate(con.comp_idc[2:end])
        a = MOI.ScalarAffineTerm(1.0, vars[comp_idx])
        rel_jumps = filter(x -> x[1] == con.ts[t_idx], con.jumps)
        if length(rel_jumps) > 0
            b_accum += con.a*rel_jumps[1][2]
        else
            push!(minus_dts_idc, t_idx)
        end
        MOI.add_constraints(
            opt,
            MOI.ScalarAffineFunction(vcat(a, minus_dts[minus_dts_idc]), -b_accum),
            MOI.EqualTo(0.0)
        )
    end
end     


struct LinearVectorLinkConstraint <: QC.LinearConstraint
    comps1::AbstractVector{Int}
    comps2::AbstractVector{Int}
    G::AbstractMatrix{Float64}
    label::String

    function LinearVectorLinkConstraint(
        name::Symbol,
        traj::NT.NamedTrajectory,
        t::Int,
        G::AbstractMatrix{Float64},
        label::String="linear vector link constraint"
    )
        comps1 = NTidx.slice(t, traj.components[name], traj.dim)
        comps2 = NTidx.slice(t+1, traj.components[name], traj.dim)
        @assert length(comps1) == length(comps2) == size(G, 1) == size(G, 2)
        return new(comps1, comps2, G, label)
    end
end

function (con::LinearVectorLinkConstraint)(
    opt::Ipopt.Optimizer,
    vars::Vector{MOI.VariableIndex},
    traj::NT.NamedTrajectory
)
    # 0 = 1 * comps2 - G * comps1
    N = length(con.comps1)

    for i=1:N
        terms = [MOI.ScalarAffineTerm(1.0, vars[con.comps2[i]])]
        for j=1:N
            if !iszero(con.G[i,j])
                push!(terms, MOI.ScalarAffineTerm(
                                -con.G[i,j],
                                vars[con.comps1[j]]
                            )
                )
            end
        end
        MOI.add_constraints(
            opt,
            MOI.ScalarAffineFunction(terms, 0.0),
            MOI.EqualTo(0.0)
        )
    end
end


function ConstantLinkConstraint(
    state_name::Symbol,
    traj::NT.NamedTrajectory,
    t::Int,
    G::Matrix{Float64}
)
    dim = traj.dims[state_name]
    zdim = traj.dim
    state_comps = traj.components[state_name]
    state_slice_t1 = NTidx.slice(t, state_comps, zdim)
    state_slice_t2 = NTidx.slice(t+1, state_comps, zdim)
    @assert size(G) == (dim, dim)

    g = Z⃗ -> Z⃗[state_slice_t2] - G * Z⃗[state_slice_t1]

    ∂g_structure = [(i, j) for i=1:dim for j in state_slice_t1]
    append!(∂g_structure, [(i, state_slice_t2[i]) for i=1:dim])
    ∂g = Z⃗ -> vcat(-vec(G'), ones(dim))

    μ∂²g_structure = []
    μ∂²g = (Z⃗, μ) -> []

    return QC.NonlinearEqualityConstraint(
        g,
        ∂g,
        ∂g_structure,
        μ∂²g,
        μ∂²g_structure,
        dim,
        Dict()
    )
end


function get_link_constraints(
    state_name::Union{Symbol, AbstractVector{Symbol}},
    traj::NT.NamedTrajectory,
    t::Int,
    G::Union{AbstractMatrix{Float64}, AbstractVector{AbstractMatrix{Float64}}},
    fix_before::NamedTuple{names, <:Tuple{Vararg{AbstractVector{Float64}}}} where names,
    fix_after::NamedTuple{names, <:Tuple{Vararg{AbstractVector{Float64}}}} where names;
    hard_equality_constraint=false
)
    if isa(state_name, Symbol)
        state_name = [state_name]
    end
    if isa(G, AbstractMatrix{Float64})
        G = fill(G, length(state_name))
    end
    @assert length(state_name) == length(G)

    if hard_equality_constraint
        link_cons = [LinearVectorLinkConstraint(sname, traj, t, g) for (sname, g) in zip(state_name, G)]
    else
        link_cons = [ConstantLinkConstraint(sname, traj, t, g) for (sname, g) in zip(state_name, G)]
    end
    eq_cons_before = [QC.EqualityConstraint(
                        [t], 
                        traj.components[name], 
                        val, 
                        traj.dim; 
                        label="fix :$name before cut at t=$t"
                        ) for (name, val) in pairs(fix_before)]
    eq_cons_after = [QC.EqualityConstraint(
                        [t+1], 
                        traj.components[name], 
                        val, 
                        traj.dim; 
                        label="fix :$name after cut at t=$t"
                        ) for (name, val) in pairs(fix_after)]
    return vcat(link_cons, eq_cons_before, eq_cons_after)
end


function custom_bounds_constraint(
    name::Symbol,
    traj::NT.NamedTrajectory,
    ts_exclude::AbstractVector{Int},
    bounds::Union{Tuple{R, R}, Vector{Tuple{R, R}}, Vector{R}, R}
) where R <: Real
    if name ∈ keys(traj.initial) && name ∈ keys(traj.final)
        ts = 2:traj.T-1
    elseif name ∈ keys(traj.initial) && !(name ∈ keys(traj.final))
        ts = 2:traj.T
    elseif name ∈ keys(traj.final) && !(name ∈ keys(traj.initial))
        ts = 1:traj.T-1
    else
        ts = 1:traj.T
    end
    ts = filter(t -> t ∉ ts_exclude, ts)
    js = traj.components[name]
    con_label = "bounds on $name"
    return QC.BoundsConstraint(ts, js, bounds, traj.dim; label=con_label)
end


function NameComponentEqualityConstraint(
    name::Symbol,
    comp::Int,
    traj::NT.NamedTrajectory,
    ts_sections::AbstractVector{AbstractVector{Int}},
    vals::AbstractVector{Float64}
)
    @assert length(ts) == length(vals)
    j = traj.components[name][comp]
    cons = [QC.EqualityConstraint(
                ts,
                j,
                val,
                traj.dim,
                "Equality constraint on comp $comp of :$name for ts=$ts"
            ) for (ts, val) in zip(ts_sections, vals)]
    return cons
end


struct LinearSincConvolutionConstraint <: QC.LinearConstraint
    comps::AbstractMatrix{Int}
    kernel::AbstractMatrix{Float64}
    label::String

    function LinearSincConvolutionConstraint(
        name::Symbol,
        dts_name::Symbol,
        traj::NT.NamedTrajectory,
        cutoff::Float64, # cutoff frequency
        ts=1:traj.T,
        label::String="linear convolution constraint"
    )
        dts = vec(traj[dts_name])[ts]
        kernel = sinc_kernel(cutoff, dts)
        comps = hcat([collect(NTidx.slice(t, Z_guess.components[name], Z_guess.dim)) for t in ts]...)
        return new(comps, kernel, label)
    end
end

function (con::LinearSincConvolutionConstraint)(
    opt::Ipopt.Optimizer,
    vars::Vector{MOI.VariableIndex},
    traj::NT.NamedTrajectory
)
    N_rows, N = size(con.comps)
    for k=1:N_rows
        vars_k = vars[con.comps[k,:]]
        for i=1:N
            terms = [MOI.ScalarAffineTerm(1.0, vars_k[i])]
            for j=1:N
                push!(terms, MOI.ScalarAffineTerm(
                                -con.kernel[i,j],
                                vars_k[j]
                            )
                )
            end
            MOI.add_constraints(
                opt,
                MOI.ScalarAffineFunction(terms, 0.0),
                MOI.EqualTo(0.0)
            )
        end
    end
end


function PhiSincConvolutionConstraint(
    IQ_name::Symbol,
    dts_name::Symbol,
    traj::NT.NamedTrajectory,
    cutoff::Float64,
    ts=1:traj.T
)
    dts = vec(traj[dts_name])[ts]
    kernel = sinc_kernel(cutoff, dts)
    A = LA.I(length(ts)) - kernel
    zdim = traj.dim
    T = traj.T

    I_comp = traj.components[IQ_name][1]
    Q_comp = traj.components[IQ_name][2]

    I_comps = [NTidx.index(t, I_comp, zdim) for t in ts]
    Q_comps = [NTidx.index(t, Q_comp, zdim) for t in ts]
    IQ_slices = [NTidx.slice(t, traj.components[IQ_name], zdim) for t in ts]

    gₜ(t, I, Q) = A[t,:]' * phi_IQ.(I, Q)
    ∂gₜ(t, I, Q) = A[t,:]' * phi_gradient.(I, Q)
    μₜ∂²gₜ(μ, t, I, Q) = μ * A[t,:]' * phi_hessian.(I, Q)

    @views function g(Z⃗)
        r = zeros(length(ts))
        for (i, t) ∈ enumerate(ts)
            I = Z⃗[I_comps]
            Q = Z⃗[Q_comps]
            r[i] = gₜ(t, I, Q)
        end
        return r
    end

    ∂g_structure = [(i, NTidx.index(t, comp, zdim)) for (i, t) in enumerate(ts) for comp in [I_comp, Q_comp]]
    @views function ∂g(Z⃗; ipopt=true)
        ∂ = SA.spzeros(length(ts), zdim * T)
        for (i, t) ∈ enumerate(ts)
            I = Z⃗[I_comps]
            Q = Z⃗[Q_comps]
            ∂[i, IQ_slices[i]] = ∂gₜ(t, I, Q)
        end
        if ipopt
            return [∂[i, j] for (i, j) in ∂g_structure]
        else
            return ∂
        end
    end

    μ∂²g_substructure = [
        (I_comp, I_comp),
        (I_comp, Q_comp),
        (Q_comp, Q_comp)
    ]
    μ∂²g_structure = [ij .+ NTidx.index(t, 0, zdim) for t in ts for ij in μ∂²g_substructure]
    @views function μ∂²g(Z⃗, μ; ipopt=true)
        μ∂² = SA.spzeros(zdim * T, zdim * T)
        for (i, t) ∈ enumerate(ts)
            I = Z⃗[I_comps]
            Q = Z⃗[Q_comps]
            μ∂²[IQ_slices[i], IQ_slices[i]] += μₜ∂²gₜ(μ[i], t, I, Q)
        end
        if ipopt
            return [μ∂²[i, j] for (i, j) in μ∂²g_structure]
        else
            return μ∂²
        end
    end

    return QC.NonlinearEqualityConstraint(
        g,
        ∂g,
        ∂g_structure,
        μ∂²g,
        μ∂²g_structure,
        length(ts),
        Dict()
    )
end
    

function IQPhiConstraint(
    IQ_name::Symbol,
    phi_name::Symbol,
    traj::NT.NamedTrajectory
)
    T = traj.T
    zdim = traj.dim
    I_comps = [NTidx.index(t, traj.components[IQ_name][1], zdim) for t=1:T]
    Q_comps = [NTidx.index(t, traj.components[IQ_name][2], zdim) for t=1:T]
    phi_comps = [NTidx.index(t, traj.components[phi_name][1], zdim) for t=1:T]

    gI(I, phi) = I - cos(phi)
    gQ(Q, phi) = Q - sin(phi)
    ∂gI(phi) = [1., sin(phi)]
    ∂gQ(phi) = [1., -cos(phi)]
    gI_phiphi(phi) = cos(phi)
    gQ_phiphi(phi) = sin(phi)

    @views function g(Z⃗)
        r = zeros(2*T)
        # first T constraints are (I,phi), second T constraints are (Q,phi)
        Threads.@threads for t=1:T
            I = Z⃗[I_comps[t]]
            Q = Z⃗[Q_comps[t]]
            phi = Z⃗[phi_comps[t]]
            r[t] = gI(I, phi)
            r[t+T] = gQ(Q, phi)
        end
        return r
    end

    ∂g_structure = [(t, comps[t]) for t=1:T for comps in [I_comps, phi_comps]]
    append!(∂g_structure, [(t+T, comps[t]) for t=1:T for comps in [Q_comps, phi_comps]])
    @assert length(∂g_structure) == 4*T
    @views function ∂g(Z⃗)
        ∂ = Array{Float64}(undef, 4*T)
        Threads.@threads for t=1:T
            phi = Z⃗[phi_comps[t]]
            ∂[2*t-1:2*t] = ∂gI(phi)
            ∂[(2*t-1:2*t) .+ 2*T] = ∂gQ(phi)
        end
        return ∂
    end

    μ∂²g_structure = [(phi_comp, phi_comp) for phi_comp in phi_comps]
    @assert length(μ∂²g_structure) == T
    @views function μ∂²g(Z⃗, μ)
        μ∂² = Array{Float64}(undef, T)
        Threads.@threads for t=1:T
            phi = Z⃗[phi_comps[t]]
            μ∂²[t] = μ[t] * gI_phiphi(phi) + μ[t+T] * gQ_phiphi(phi)
        end
        return μ∂²
    end

    return QC.NonlinearEqualityConstraint(
        g,
        ∂g,
        ∂g_structure,
        μ∂²g,
        μ∂²g_structure,
        2*T,
        Dict()
    )
end


function FinalYZGreatCircleConstraint(
    state_name::Symbol,
    comps::AbstractVector{Int64},
    traj::NT.NamedTrajectory
)
    @assert length(comps) == 2
    T = traj.T
    zdim = traj.dim
    offset = NTidx.index(T, 0, zdim)
    state_dim = div(traj.dims[state_name], 2) # because iso
    comps = vcat(
        comps .+ offset,
        comps .+ offset .+ state_dim
    )
    gx(x) = x[1]*x[2] + x[3]*x[4]
    ∂gx(x) = [x[2], x[1], x[4], x[3]]

    g(Z⃗) = [gx(Z⃗[comps])]

    ∂g_structure = [(1, comps[i]) for i=1:4]
    ∂g(Z⃗) = ∂gx(Z⃗[comps])

    μ∂²g_structure = []
    μ∂²g(Z⃗, μ) = []

    return QC.NonlinearEqualityConstraint(
        g,
        ∂g,
        ∂g_structure,
        μ∂²g,
        μ∂²g_structure,
        1,
        Dict()
    )
end