import QuantumCollocation as QC
import NamedTrajectories as NT
import TrajectoryIndexingUtils as NTidx
import ForwardDiff as FD


function NameComponentPathObjective(
    name::Symbol,
    comps::AbstractVector{Int},
    ts::AbstractVector{Int},
    f::AbstractVector, # need to be functions describing paths
    g::AbstractVector, # need to be functions describing mapping of components
    R::AbstractMatrix{Float64};
    is_iso=false
)
    params = Dict(
        :type => :NameComponentPathObjective,
        :name => name,
        :comps => comps,
        :R => R,
        :is_iso => is_iso
    )

    num_paths = length(f)
    @assert num_paths == length(comps) == length(g)

    if is_iso
        g_til = [x -> g[i](QC.iso_to_ket(x)[1]) for i=1:num_paths]
        dg = [x -> FD.gradient(g_til[i], x) for i=1:num_paths]
        ddg = [x -> FD.hessian(g_til[i], x) for i=1:num_paths]
    else
        g_til = g
        dg = [x -> FD.derivative(g[i], x) for i=1:num_paths]
        ddg = [x -> FD.derivative(dg[i], x) for i=1:num_paths]
    end
    df = [time -> FD.derivative(f[i], time) for i=1:num_paths]
    ddf = [time -> FD.derivative(df[i], time) for i=1:num_paths]

    function get_comp_idc_traj(name_slice, comp_idx)
        name_dim = length(name_slice)
        comp = comps[comp_idx]
        if is_iso
            return name_slice[[comp, comp + div(name_dim, 2)]]
        else
            return name_slice[comp]
        end
    end

    @views function L(Z⃗::AbstractVector{<:Real}, Z::NT.NamedTrajectory)
        loss = 0
        dts_slice = [NTidx.index(t, Z.components[Z.timestep][1], Z.dim) for t=1:maximum(ts)]
        dts = Z⃗[dts_slice]
        for (i_t, t) in enumerate(ts)
            time = sum(dts[1:t]) - dts[t]
            name_slice = NTidx.slice(t, Z.components[name], Z.dim)
            for i=1:num_paths
                x = Z⃗[get_comp_idc_traj(name_slice, i)]
                h_it = g_til[i](x) - f[i](time)
                loss += 0.5 * R[i, i_t] * h_it^2
            end
        end
        return loss
    end

    @views function ∇L(Z⃗::AbstractVector{<:Real}, Z::NT.NamedTrajectory)
        ∇ = SA.spzeros(Z.dim * Z.T)
        dts_slice = [NTidx.index(t, Z.components[Z.timestep][1], Z.dim) for t=1:maximum(ts)]
        dts = Z⃗[dts_slice]
        for (i_t, t) in enumerate(ts)
            time = sum(dts[1:t]) - dts[t]
            name_slice = NTidx.slice(t, Z.components[name], Z.dim)
            for i=1:num_paths
                comp_idc = get_comp_idc_traj(name_slice, i)
                x = Z⃗[comp_idc]
                h_it = g_til[i](x) - f[i](time)
                ∇[comp_idc] += R[i, i_t] * h_it * dg[i](x)
                ∇[dts_slice[1:t-1]] .+= -R[i, i_t] * h_it * df[i](time)
            end
        end
        return ∇
    end

    function ∂²L_structure(Z::NT.NamedTrajectory)
        dts_slice = [NTidx.index(t, Z.components[Z.timestep][1], Z.dim) for t=1:maximum(ts)]
        structure = [(idx, idx) for idx in dts_slice]
        for t in ts
            name_slice = NTidx.slice(t, Z.components[name], Z.dim)
            for i=1:num_paths
                comp_idc = get_comp_idc_traj(name_slice, i)
                append!(structure, [(comp_idc[i], comp_idc[j]) for i=1:length(comp_idc) for j=1:i])
            end
        end
        return structure
    end

    @views function ∂²L(Z⃗::AbstractVector{<:Real}, Z::NT.NamedTrajectory; return_moi_vals=true)
        H = SA.spzeros(Z.dim * Z.T, Z.dim * Z.T)
        dts_slice = [NTidx.index(t, Z.components[Z.timestep][1], Z.dim) for t=1:maximum(ts)]
        dts = Z⃗[dts_slice]
        for (i_t, t) in enumerate(ts)
            time = sum(dts[1:t]) - dts[t]
            name_slice = NTidx.slice(t, Z.components[name], Z.dim)
            for i=1:num_paths
                comp_idc = get_comp_idc_traj(name_slice, i)
                x = Z⃗[comp_idc]
                h_it = g_til[i](x) - f[i](time)
                dg_it = dg[i](x)
                H[comp_idc, comp_idc] += R[i, i_t] * (dg_it * dg_it' + h_it * ddg[i](x))
                H[dts_slice[1:t-1], dts_slice[1:t-1]] .+= R[i, i_t] * (df[i](time)^2 - h_it * ddf[i](time))
            end
        end
        if return_moi_vals
            Hs = [H[i,j] for (i, j) ∈ ∂²L_structure(Z)]
            return Hs
        else
            return H
        end
    end

    return QC.Objective(L, ∇L, ∂²L, ∂²L_structure, Dict[params])
end


function NameComponentObjective(
    name::Symbol,
    comps::AbstractVector{Int},
    ts::AbstractVector{Int},
    f::Function, # C^N -> R
    R::AbstractVector{Float64}=ones(length(ts));
    is_iso=false
)
    params = Dict(
        :type => :StateComponentObjective,
        :name => name,
        :comps => comps,
        :R => R,
        :eval_hessian => true,
        :is_iso => is_iso
    )

    num_comps = is_iso ? length(comps) * 2 : length(comps)

    # R^(2)N -> R
    if is_iso
        f_til = x_til -> f(QC.iso_to_ket(x_til)) 
    else
        f_til = f
    end
    df_til = x_til -> FD.gradient(f_til, x_til) # R^(2)N -> R^(2)N
    ddf_til = x_til -> FD.hessian(f_til, x_til) # R^(2)N -> R^{(2)N x (2)N}
    
    function get_comp_idc_traj(name_slice)
        name_dim = length(name_slice)
        idc = copy(comps)
        if is_iso
            append!(idc, comps .+ div(name_dim, 2))
        end
        return name_slice[idc]
    end

    @views function L(Z⃗::AbstractVector{<:Real}, Z::NT.NamedTrajectory)
        l = 0
        for (r, t) in zip(R, ts)
            name_slice = NTidx.slice(t, Z.components[name], Z.dim)
            x_til = Z⃗[get_comp_idc_traj(name_slice)]
            l += r * f_til(x_til)
        end
        return l
    end

    @views function ∇L(Z⃗::AbstractVector{<:Real}, Z::NT.NamedTrajectory)
        ∇ = SA.spzeros(Z.dim * Z.T)
        for (r, t) in zip(R, ts)
            name_slice = NTidx.slice(t, Z.components[name], Z.dim)
            comp_idc = get_comp_idc_traj(name_slice)
            ∇[comp_idc] = r * df_til(Z⃗[comp_idc])
        end
        return ∇
    end

    function ∂²L_structure(Z::NT.NamedTrajectory)
        structure = []
        for t in ts
            name_slice = NTidx.slice(t, Z.components[name], Z.dim)
            comp_idc = get_comp_idc_traj(name_slice)
            append!(structure, [(comp_idc[i], comp_idc[j]) for i=1:num_comps for j=1:i])
        end
        return structure
    end

    @views function ∂²L(Z⃗::AbstractVector{<:Real}, Z::NT.NamedTrajectory; return_moi_vals=true)
        H = SA.spzeros(Z.dim * Z.T, Z.dim * Z.T)
        for (r, t) in zip(R, ts)
            name_slice = NTidx.slice(t, Z.components[name], Z.dim)
            comp_idc = get_comp_idc_traj(name_slice)
            H[comp_idc, comp_idc] = r * ddf_til(Z⃗[comp_idc])
        end
        if return_moi_vals
            Hs = [H[i,j] for (i, j) ∈ ∂²L_structure(Z)]
            return Hs
        else
            return H
        end
    end

    return QC.Objective(L, ∇L, ∂²L, ∂²L_structure, Dict[params])
end


function NameComponentQuadraticRegularizer(
    name::Symbol,
    comps::AbstractVector{Int},
    traj::NT.NamedTrajectory,
    R::AbstractVector{<:Real};
    is_iso=false
)
    @assert length(comps) == length(R)
    name_dims = traj.dims[name]
    R_full = zeros(name_dims)
    R_full[comps] = R
    if is_iso
        R_full[comps .+ div(name_dims, 2)] = R
    end
    return QC.QuadraticRegularizer(
        name,
        traj,
        R_full
    )
end




function steer_to_pops(pops)
    return x -> sum(abs2.(abs2.(x) .- pops))
end



function QuadraticObjective(
    name::Symbol,
    traj::NT.NamedTrajectory,
	R::AbstractMatrix{<:Real},
    comp_idc::AbstractArray{Int}=1:traj.dims[name],
    ts::AbstractVector{Int}=[traj.T];
    Q::Float64=1.0,
    time_major::Bool=true
)
    params = Dict(
        :type => :QuadraticObjective,
        :name => name,
        :comp_idc => comp_idc,
        :ts => ts,
        :R => R,
        :Q => Q
    )

    if time_major
        name_slice = [NTidx.index(t, traj.components[name][i], traj.dim) for i in comp_idc for t in ts]
    else
        name_slice = [NTidx.index(t, traj.components[name][i], traj.dim) for t in ts for i in comp_idc]
    end

    full_dim = length(name_slice)
    @assert size(R) == (full_dim, full_dim)

    A = 0.5 * (R + R')

	@views function L(Z⃗::AbstractVector{<:Real}, Z::NT.NamedTrajectory)
		v = Z⃗[name_slice]
		return Q * 0.5 * v' * R * v
	end

	@views function ∇L(Z⃗::AbstractVector{<:Real}, Z::NT.NamedTrajectory)
		∇ = zeros(Z.dim * Z.T)
        v = Z⃗[name_slice]
		∇[name_slice] = Q * A * v
		return ∇
	end

    A_is, A_js, _ = SA.findnz(SA.sparse(A))
    A_structure = collect(zip(A_is, A_js))
    filter!(ij -> ij[1] >= ij[2], A_structure)
    ∂²L_structure = _ -> [(name_slice[i], name_slice[j]) for (i, j) in A_structure]
    # function ∂²L_structure(Z::NT.NamedTrajectory)
    #     offset = NTidx.index(Z.T, Z.components[name][1], Z.dim) - 1
    #     structure = [
    #         ij .+ offset for ij in A_structure
    #     ]
    #     return structure
    # end

	@views function ∂²L(Z⃗::AbstractVector{<:Real}, Z::NT.NamedTrajectory; return_moi_vals=true)
        H = SA.spzeros(Z.dim * Z.T, Z.dim * Z.T)
        H[name_slice, name_slice] = Q * A
        if return_moi_vals
            Hs = [H[i,j] for (i, j) ∈ ∂²L_structure(Z)]
            return Hs
        else
            return H
        end
    end

	return QC.Objective(L, ∇L, ∂²L, ∂²L_structure, Dict[params])
end