import NamedTrajectories as NT
import LinearAlgebra as LA
import SparseArrays as SA
import Interpolations as IP

"""
Fourier transform quantum state from position into momentum space. Assumes discrete integer momenta in [-p_max, p_max].
Output state is normalized if input state is.
"""
function x_to_p(
    psi_x::Vector{<:Number}, 
    xs::Vector{<:Real}, 
    p_max::Int
)
    ps = collect(-p_max:p_max) 
    ft_mat = exp.(-1im*ps*xs')
    return ft_mat * psi_x .* (xs[2] - xs[1]) / sqrt.(2pi)
end


"""
Fourier transform quantum state from momentum into position space. Assumes discrete integer momenta in [-p_max, p_max].
Output state is normalized if input state is.
"""
function p_to_x(
    psi_p::Vector{<:Number}, 
    xs::Vector{<:Real}
)
    dim = length(psi_p)
    p_max = div(dim - 1, 2)
    ps = collect(-p_max:p_max) 
    ft_mat = exp.(1im*xs*ps')
    return ft_mat * psi_p / sqrt.(2pi)
end


function area(
    y::Vector{<:Real},
    dx::Vector{<:Real}
)
    return sum((y[2:end] + y[1:end-1])/2 .* dx[1:end-1])
end

function area(
    y::Vector{<:Real},
    dx::Real
)
    return sum((y[2:end] + y[1:end-1]))/2 * dx
end

function gaussian_state(xs, s)
    dxs = xs[2:end] - xs[1:end-1]
    push!(dxs, 0.)
    psi = exp.(-xs.^2/s^2)
    psi ./= sqrt(area(abs2.(psi), dxs))
    return psi
end

function normalize!(psi)
    psi ./= LA.norm(psi)
end

function blockdiagonal(mats...)
    dims = [size(mat) for mat in mats]
    height = sum([d[1] for d in dims])
    width = sum([d[2] for d in dims])
    B = SA.spzeros(eltype(mats[1]), height, width)
    i = 1
    j = 1
    for (mat, d) in zip(mats, dims)
        B[i:i+d[1]-1, j:j+d[2]-1] = mat
        i += d[1]
        j += d[2]
    end
    return B
end

function blockmatrix(height, width, mats::Dict{Tuple{Int, Int}, <:AbstractMatrix}; type=Float64)
    @assert maximum([ij[1]+size(mat, 1)-1 for (ij, mat) in mats]) <= height
    @assert maximum([ij[2]+size(mat, 2)-1 for (ij, mat) in mats]) <= width
    B = SA.spzeros(type, height, width)
    for (ij, mat) in mats
        d = size(mat)
        @assert iszero(B[ij[1]:ij[1]+d[1]-1, ij[2]:ij[2]+d[2]-1])
        B[ij[1]:ij[1]+d[1]-1, ij[2]:ij[2]+d[2]-1] = mat
    end
    return B
end


"""
Fourier transform (x, f(x)) -> (k, f̃(k))
f̃(k) = 1/√2π ∫ f(x) exp(±ikx) dx
"""
function fourier(fx, xs, ks; exp_sign=+1.)
    dxs = xs[2:end] - xs[1:end-1]
    push!(dxs, dxs[end])
    fx_dxs_sqrt2pi = fx .* dxs / sqrt(2pi)
    fk = [exp.(exp_sign*1im*k*xs)' * fx_dxs_sqrt2pi for k in ks]
    return fk
end


function interpolate_controls(
    controls::Matrix{Float64},
    dts_old::Vector{Float64},
    dts_new::Vector{Float64}
)
    times_old = cumsum(dts_old) - dts_old
    duration = times_old[end]
    T_old = length(dts_old)
    times_new = cumsum(dts_new) - dts_new
    @assert duration >= times_new[end] - 1e-8

    ts_new = []
    t_old = 1
    time_old1 = times_old[t_old]
    time_old2 = times_old[t_old+1]
    for time_new in times_new
        while true
            if time_old1 <= time_new < time_old2
                t_new = t_old + (time_new - time_old1) / (time_old2 - time_old1)
                push!(ts_new, t_new)
                break
            elseif isapprox(time_new, duration)
                push!(ts_new, T_old)
                break
            else
                t_old += 1
                time_old1 = times_old[t_old]
                time_old2 = times_old[t_old+1]
            end
        end
    end

    controls_itp = IP.interpolate(controls, (IP.NoInterp(), IP.BSpline(IP.Cubic(IP.Free(IP.OnCell())))))
    controls_new = collect(hcat([[controls_itp(j,t_new) for t_new in ts_new] for j=1:size(controls,1)]...)')
    return controls_new
end


function sinc_kernel(
    cutoff::Float64,
    dts::Vector{Float64}
)
    T = length(dts)
    times = cumsum(dts) - dts
    timestimes = times * ones(T)' - ones(T) * times'
    kernel = cutoff/pi * sinc.((cutoff/pi)*timestimes) .* dts' # for some reason extra 1/pi factor?
    return kernel
end

phi_IQ(I, Q) = atan(Q, I)

function phi_gradient(
    I::Float64,
    Q::Float64
)
    # assuming I^2 + Q^2 = 1
    return [-Q, I]
end

function phi_hessian(
    I::Float64,
    Q::Float64
)
    # assuming I^2 + Q^2 = 1
    x = 2*I*Q
    y = -1 + 2*Q^2
    return [x y; y -x]
end


function trajectory_shrink_extend(
    Z::NT.NamedTrajectory,
    T_new::Int64
)
    if T_new < Z.T
        data = [Z[name][:,1:T_new] for name in Z.names]
    elseif T_new > Z.T
        data = [
            hcat(Z[name], repeat(Z[name][:,Z.T], 1, T_new-Z.T)) 
            for name in Z.names]
    else
        return Z
    end
    comps = (; zip(Z.names, data)...)
    return NT.NamedTrajectory(
        comps;
        controls=Z.control_names,
        timestep=Z.timestep,
        bounds=Z.bounds,
        initial=Z.initial,
        final=Z.final,
        goal=Z.goal
    )
end