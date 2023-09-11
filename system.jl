import QuantumCollocation as QC
import LinearAlgebra as LA

include("utils.jl")

"""
ħ = 1

Hamiltonian in real space:
H = p̂²/2m - V₀/2 cos(2κx + φ(t))
where p̂ = 1/i d/dx

Hamiltonian in momentum space:
H = p̂²/2m - V₀/4 (e^{iφ} a_{2κ} + e^{-iφ} a†_{2κ})
where a_{2κ} lowers momentum by 2κ

Define lattice recoil energy E_R = ħκ²/2m (in Hz),
and redefine p̂ -> 2κ (p̂ + p₀). p̂ is integer-valued and 
counts the momentum quanta in units of 2κ.

Final Hamiltonian, in units of E_R:
H = 4 (p̂ + p₀)² - V/4 (e^{iφ} a + e^{-iφ} a†)
H = 4 (p̂ + p₀)² - V/4 (I(t) (a + a†) + iQ(t) (a - a†))

Note: a (a†) destroy/create momentum quanta (p̂ +- 1).
QOC task: Optimize for shaking protocol φ(t). However only
linear Hamiltonian structure available in QuantumCollocation,
thus optimize for controls I(t) and Q(t) such that
e^{iφ} = I + iQ.
Necessary: Non-linear equality constraint I²(t) + Q²(t) = 1.

Basis in momentum space:
|-p_max + p₀⟩ = [1 0 ... 0]'
|-p_max + 1 + p₀⟩ = [0 1 0 ... 0]'
...
|p₀⟩ = [0 ... 0 1(middle index) 0 ... 0]'
...
|p_max + p₀⟩ = [0 ... 0 1]'
"""
hbar = 1.054571817e-34 # Js
function ShakenLatticeSystem1D(
    V::Float64, # lattice potential, units of E_R
    trunc::Int; # momentum or Bloch truncation, units of 2kappa
    mass=3.82e-26, # kg
    lambda_lattice=985e-9, # m
    kappa=2π/lambda_lattice, # 2pi/m
    E_R=hbar*kappa^2/(2*mass)/1000/(2π), # lattice recoil energy/frequency, kHz
    bloch_basis=false,
    bloch_transformation_trunc=trunc, # number of Bloch states to compute before truncation
    odd_even_fix_above_degeneracy_diff=1e-20, # change Bloch states to (odd, even) pairs if energy degeneracy < diff
    acc=0., # acceleration, π/kappa * E_R^2
    include_acc_derivative=false,
    include_V_derivative=false,
    p0=0., # units of 2kappa
    sparse=false
)
    accelerated = (abs(acc) > 1e-20) || include_acc_derivative
    if bloch_basis
        dim = bloch_transformation_trunc
        mid = 0
    else
        dim = 2*trunc + 1
        mid = div(dim + 1, 2)
    end
    p_max = div(dim, 2)

    # dense
    p = LA.diagm(range(-p_max, p_max) .+ p0)
    # lower momentum by 1 (units of 2kappa)
    lower = LA.diagm(1 => ones(Int, 2*p_max))

    _H_drift = 4 * p.^2
    _H_drives = -V / 4 .* [
        lower + lower',
        1im * (lower - lower')
    ]

    H_0 = _H_drift + _H_drives[1]
    F = LA.eigen(collect(H_0))
    bloch_energies = F.values
    bloch_states = F.vectors
    for i=1:2*p_max
        if abs(bloch_energies[i] - bloch_energies[i+1]) < odd_even_fix_above_degeneracy_diff
            odd_before = bloch_states[:,i]
            even_before = bloch_states[:,i+1]
            even_signs = sign.(real.(even_before))
            even_signs[end-p_max+1:end] = even_signs[1:p_max]
            pops = abs2.(even_before + 1im*odd_before)/2
            even = sqrt.(pops) .* even_signs
            odd = copy(even)
            odd[1:p_max] *= -1
            bloch_states[:,i] = odd
            bloch_states[:,i+1] = even
        end
    end

    if bloch_basis
        # transform and truncate
        p = (bloch_states' * p * bloch_states)[1:trunc,1:trunc]
        _H_drift = (bloch_states' * _H_drift * bloch_states)[1:trunc,1:trunc]
        _H_drives = [(bloch_states' * Hd * bloch_states)[1:trunc,1:trunc] for Hd in _H_drives]
        dim = trunc
    end

    if accelerated
        H_acc = Matrix{Float64}[
            2pi * acc * p
        ]
        append!(_H_drives, H_acc)
    end

    H_drift = copy(_H_drift)
    H_drives = copy(_H_drives)
    ham_dim = dim

    if include_acc_derivative
        ham_dim += dim
        H_drift = blockdiagonal(H_drift, _H_drift)
        H_drives = [blockdiagonal(Hd, _Hd) for (Hd, _Hd) in zip(H_drives, _H_drives)]
        H_drives[end] += blockmatrix(ham_dim, ham_dim, Dict((ham_dim-dim+1, 1) => 2pi * p))
    end

    if include_V_derivative
        ham_dim += dim
        H_drift = blockdiagonal(H_drift, _H_drift)
        H_drives = [blockdiagonal(Hd, _Hd) for (Hd, _Hd) in zip(H_drives, _H_drives)]
        H_drives[1] += blockmatrix(ham_dim, ham_dim, Dict((ham_dim-dim+1, 1) => _H_drives[1]/V))
        H_drives[2] += blockmatrix(ham_dim, ham_dim, Dict((ham_dim-dim+1, 1) => _H_drives[2]/V); type=ComplexF64)
    end

    if sparse
        H_drift = SA.sparse(H_drift)
        H_drives = SA.sparse.(H_drives)
    else
        H_drift = collect(H_drift)
        H_drives = collect.(H_drives)
    end

    params = Dict(
        :type => :ShakenLatticeSystem1D,
        :V => V,
        :trunc => trunc,
        :dim => dim,
        :mid => mid,
        :E_R => E_R,
        :bloch_basis => bloch_basis,
        :bloch_energies => bloch_energies,
        :bloch_states => bloch_states,
        :acc => acc,
        :accelerated => accelerated,
        :p0 => p0
    )
    
    return QC.QuantumSystem(
        H_drift, 
        H_drives; 
        params=params
    )
end


function get_bloch_state(
    system::QC.QuantumSystem,
    phi=0.,
    I=cos(phi),
    Q=sin(phi);
    lvl::Int=0
)
    @assert system.params[:type] == :ShakenLatticeSystem1D

    H = system.H_drift_real + 1im*system.H_drift_imag
    H += I * (system.H_drives_real[1] + 1im*system.H_drives_imag[1])
    H += Q * (system.H_drives_real[2] + 1im*system.H_drives_imag[2])
    if system.params[:accelerated]
        dim = system.params[:dim]
        H = H[1:dim,1:dim]
    end
    psi = LA.eigvecs(collect(H))[:,lvl+1] # is normalized
    return psi
end


function get_shaken_lattice_propagator(
    system::QC.QuantumSystem,
    time1::Float64,
    time0::Float64=0.0,
    T::Int=2;
    return_iso_operator=true
)
    @assert system.params[:type] == :ShakenLatticeSystem1D
    controls = vcat(ones(T)', zeros(T)')
    dts = fill((time1 - time0)/(T-1), T)
    if system.params[:accelerated]
        times = cumsum(dts) - dts .+ time0
        controls = vcat(controls, times')
    end
    U_iso = QC.unitary_rollout(controls, dts, system; integrator=exp)[:,T]
    if return_iso_operator
        return QC.iso_vec_to_iso_operator(U_iso)
    else
        return QC.iso_vec_to_operator(U_iso)
    end
end


function get_shaken_lattice_propagator(
    system::QC.QuantumSystem,
    times::AbstractVector{Float64},
    jumps::AbstractVector{Tuple{Int, Float64}},
    jump_Ts::Union{Int, AbstractVector{Int}}=2;
    return_iso_operator=true
)
    if isa(jump_Ts, Int)
        jump_Ts = fill(jump_Ts, length(jumps))
    end
    @assert length(jump_Ts) == length(jumps)
    G = Matrix{Float64}[]
    offset = 0
    for ((cut, jump_time), jump_T) in zip(jumps, jump_Ts)
        push!(G, get_shaken_lattice_propagator(
                    system, 
                    times[cut] + jump_time + offset, 
                    times[cut] + offset, 
                    jump_T;
                    return_iso_operator=return_iso_operator)
            )
        offset += jump_time 
    end
    return G
end


function get_times(
    dts::AbstractVector{Float64},
    jumps::AbstractVector{Tuple{Int, Float64}}=Tuple{Int, Float64}[]
)
    times_slices = []
    i = 1
    time_offset = 0.
    for (cut, jump_time) in jumps
        dts_slice = dts[i:cut]
        times = cumsum(dts_slice) - dts_slice .+ time_offset
        push!(times_slices, times)
        i = cut + 1
        time_offset = times[end] + jump_time
    end
    dts_slice = dts[i:end]
    times = cumsum(dts_slice) - dts_slice .+ time_offset
    push!(times_slices, times)
    return vcat(times_slices...)
end



function shaken_lattice_rollout(
    psi0_iso::AbstractVector{Float64},
    controls::Matrix{Float64},
    dts::AbstractVector{Float64},
    system::QC.QuantumSystem,
    jumps::AbstractVector{Tuple{Int, Float64}}=Tuple{Int, Float64}[],
    jump_propagators::AbstractVector{Matrix{Float64}}=Matrix{Float64}[];
    rollout_kwargs...
)
    @assert system.params[:type] == :ShakenLatticeSystem1D
    if system.params[:accelerated]
        times = get_times(dts, jumps)
        controls = vcat(controls, times')
    end
    psi_isos = []
    i = 1
    for ((cut, _), G) in zip(jumps, jump_propagators)
        psi_iso = QC.rollout(psi0_iso, controls[:,i:cut], dts[i:cut], system; rollout_kwargs...)
        push!(psi_isos, psi_iso)
        i = cut + 1
        psi0_iso = G * psi_iso[:,end]
    end
    psi_iso = QC.rollout(psi0_iso, controls[:,i:end], dts[i:end], system; rollout_kwargs...)
    push!(psi_isos, psi_iso)
    return hcat(psi_isos...)
end


function get_controls_dts(
    controls::AbstractMatrix{Float64},
    dts::AbstractVector{Float64},
    jumps::AbstractVector{Tuple{Int, Float64}}=Tuple{Int, Float64}[],
    jump_Ts::Union{Int, AbstractVector{Int}}=Int[]
)
    if isa(jump_Ts, Int)
        jump_Ts = fill(jump_Ts, length(jumps))
    end
    @assert length(jumps) == length(jump_Ts)
    controls_new = zeros(size(controls,1), 0)
    dts_new = zeros(0)
    i = 1
    for ((cut, jump_time), jump_T) in zip(jumps, jump_Ts)
        controls_jump = vcat(ones(jump_T-1)', zeros(jump_T-1)')
        controls_new = hcat(controls_new, controls[:,i:cut-1], controls_jump)
        dts_jump = fill(jump_time/(jump_T-1), jump_T-1)
        dts_new = vcat(dts_new, dts[i:cut-1], dts_jump)
        i = cut + 1
    end
    controls_new = hcat(controls_new, controls[:,i:end])
    dts_new = vcat(dts_new, dts[i:end])
    return controls_new, dts_new
end


function shaken_lattice_rollout(
    psi0_iso::AbstractVector{Float64},
    controls::Matrix{Float64},
    dts::AbstractVector{Float64},
    system::QC.QuantumSystem,
    jumps::AbstractVector{Tuple{Int, Float64}}=Tuple{Int, Float64}[],
    jump_Ts::Union{Int, AbstractVector{Int}}=Int[];
    rollout_kwargs...
)
    controls_full, dts_full = get_controls_dts(controls, dts, jumps, jump_Ts)
    if system.params[:accelerated]
        times = cumsum(dts_full) - dts_full
        controls_full = vcat(controls_full, times')
    end
    return QC.rollout(psi0_iso, controls_full, dts_full, system; rollout_kwargs...)
end


function phi_mod_clean!(phi)
    delta = 1.0
    for i=2:length(phi)
        if phi[i] - phi[i-1] > delta
            phi[i:end] .-= 2pi
        elseif phi[i] - phi[i-1] < -delta
            phi[i:end] .+= 2pi
        end
    end
end


function get_interferometer(Z_split, Z_mirror, dts_flight)
    a_split = Z_split.a[:,1:end-1]
    a_mirror = Z_mirror.a[:,1:end-1]
    @assert size(a_split, 1) == size(a_mirror, 1)
    T_flight = length(dts_flight)
    T_full = 2*(Z_split.T-1) + (Z_mirror.T-1) + 2*T_flight + 1

    a_full = zeros(size(a_split, 1), T_full)
    a_full[1,:] .+= 1.
    dts_full = zeros(T_full)
    start = 1

    a_full[:,start:start-1+Z_split.T-1] = a_split
    dts_full[start:start-1+Z_split.T-1] = vec(Z_split.dts)[1:end-1]
    start += Z_split.T - 1

    dts_full[start:start-1+T_flight] = vec(dts_flight)
    start += T_flight

    a_full[:,start:start-1+Z_mirror.T-1] = a_mirror
    dts_full[start:start-1+Z_mirror.T-1] = vec(Z_mirror.dts)[1:end-1]
    start += Z_mirror.T - 1

    dts_full[start:start-1+T_flight] = vec(dts_flight)
    start += T_flight

    a_full[:,start:end-1] = a_split[:,end:-1:1]
    dts_full[start:end-1] = vec(Z_split.dts)[end-1:-1:1]
    return a_full, dts_full
end
