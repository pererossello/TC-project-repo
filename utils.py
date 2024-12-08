import numpy as np
import pymaster as nmt


def compute_master(f_a, f_b, wsp):
    cl_coupled = nmt.compute_coupled_cell(f_a, f_b)
    cl_decoupled = wsp.decouple_cell(cl_coupled)
    return cl_decoupled


def auto_spectrum(
    mask, map, lmax, dl, nside, purify_e=False, purify_b=False, beam=None
):
    """
    Computes Power Spectra for each polarization mode, intensity, and cross modes
    """

    #Create field spin0
    f0 = nmt.NmtField(mask, [map[0, :]], lmax_sht=lmax, beam=beam)
    # Create field spin2
    f2 = nmt.NmtField(
        mask,
        [map[1, :], map[2, :]],
        lmax_sht=lmax,
        purify_e=purify_e,
        purify_b=purify_b,
        beam=beam,
    )

    # Create binning scheme
    b = nmt.NmtBin(nside, nlb=dl, lmax=lmax)
    w00 = nmt.NmtWorkspace()
    w00.compute_coupling_matrix(f0, f0, b)

    w02 = nmt.NmtWorkspace()
    w02.compute_coupling_matrix(f0, f2, b)

    w22 = nmt.NmtWorkspace()
    w22.compute_coupling_matrix(f2, f2, b)

    # Compute the power spectrum of our two input fields
    cl_master_tt = compute_master(f0, f0, w00)  # TT
    cl_master_tetb = compute_master(f0, f2, w02)  # TE TB
    cl_master_eb = compute_master(f2, f2, w22)  # EE EB BE BB

    cl_tt = cl_master_tt[0]  # label='TT '
    cl_te = cl_master_tetb[0]  # label='TE '
    cl_tb = cl_master_tetb[1]  # label='TB '
    cl_ee = cl_master_eb[0]  # label='EE '
    cl_eb = cl_master_eb[1]  # label='EB '
    cl_bb = cl_master_eb[3]  # label='BB '

    power_spectrum_modes = np.array(
        [b.get_effective_ells(), cl_tt, cl_ee, cl_bb, cl_te, cl_tb, cl_eb]
    )

    return power_spectrum_modes  # ell, TT, EE, BB, TE, TB, EB
