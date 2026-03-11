import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.integrate import quad
from scipy.optimize import minimize_scalar, curve_fit
from astropy.cosmology import FlatLambdaCDM
from gbm import test_data_dir
from gbm.data import Cspec, GbmDetectorCollection, TTE, RSP
from gbm.background import BackgroundFitter
from gbm.background.binned import Polynomial
from gbm.binning.unbinned import bin_by_time
from gbm.spectra.fitting import SpectralFitterCstat
from funciones import FixedPowerLaw, FixedComptonized, BlackBody
from gbm.plot import Lightcurve, Spectrum

def cargar_configuracion_grb(nombre_grb, data="GRBs_data.dat"):
    df = pd.read_csv(data, dtype={'nombre_grb': str})

    filtro = df['nombre_grb'] == nombre_grb
    if not filtro.any():
        raise ValueError(f"GRB '{nombre_grb}' no encontrado en el archivo {data}")

    fila = df[filtro].iloc[0]

    T_start = fila['T_start']
    T_90 = fila['T_90']
    tte_files = fila['tte_files'].split(";")
    rsp_files = fila['rsp_files'].split(";")

    return T_start, T_90, tte_files, rsp_files

def procesar_fondo(nombre_grb, orden=1, bin_size=0.512,
                   erange_nai=(8.0, 900.0), erange_bgo=(650, 35000.0),
                   data="GRBs_data.dat"):

    T_start, T_90, tte_files, rsp_files = cargar_configuracion_grb(nombre_grb, data)

    view_range = (T_start - 2.0, T_start + T_90 + 1.0)
    bkgd_range = [
        (T_start - 40., T_start - 10.),
        (T_start + T_90 + 20., T_start + T_90 + 250.)
    ]

    tte_data = [TTE.open(f) for f in tte_files]
    tte_data = [tte.to_phaii(bin_by_time, bin_size, time_ref=0.0) for tte in tte_data]
    cspecs = GbmDetectorCollection.from_list(tte_data)

    print("Calculando fondo...")
    backfitters = [BackgroundFitter.from_phaii(cspec, Polynomial, time_ranges=bkgd_range) for cspec in cspecs]
    backfitters = GbmDetectorCollection.from_list(backfitters, dets=cspecs.detector())
    backfitters.fit(order=orden)

    bkgds = backfitters.interpolate_bins(cspecs.data()[0].tstart, cspecs.data()[0].tstop)
    bkgds = GbmDetectorCollection.from_list(bkgds, dets=cspecs.detector())

    background_file = f"{nombre_grb}_fondo_orden{orden}.pkl"
    with open(background_file, 'wb') as f:
        pickle.dump(bkgds, f)

    rsps = GbmDetectorCollection.from_list([RSP.open(f) for f in rsp_files])
    
    return cspecs, bkgds, rsps


def plot_curvas_y_espectros2(cspecs, bkgds, nombre_grb, data="GRBs_data.dat",
                            erange_nai=(8.0, 900.0), erange_bgo=(650, 35000.0)):
    """Genera y guarda curvas de luz y espectros por detector."""
    T_start, T_90, *_ = cargar_configuracion_grb(nombre_grb, data)
    view_range = (T_start - 5.0, T_start + T_90 + 10.0)
    src_range = (T_start, T_start + T_90)

    data_lcs = cspecs.to_lightcurve(nai_kwargs={'energy_range': erange_nai}, bgo_kwargs={'energy_range': erange_bgo})
    bkgd_lcs = bkgds.integrate_energy(nai_args=erange_nai, bgo_args=erange_bgo)
    src_lcs = cspecs.to_lightcurve(time_range=src_range, nai_kwargs={'energy_range': erange_nai}, bgo_kwargs={'energy_range': erange_bgo})

    data_specs = cspecs.to_spectrum(time_range=src_range)
    bkgd_specs = bkgds.integrate_time(*src_range)
    src_specs = cspecs.to_spectrum(time_range=src_range, nai_kwargs={'energy_range': erange_nai}, bgo_kwargs={'energy_range': erange_bgo})

    lcplots = []
    specplots = []

    for i, (data_lc, bkgd_lc, src_lc) in enumerate(zip(data_lcs, bkgd_lcs, src_lcs)):
        fig = plt.figure()
        lcplot = Lightcurve(data=data_lc, background=bkgd_lc, figure=fig)
        lcplot.add_selection(src_lc)
        lcplot.xlim = view_range

        detector = cspecs.detector()[i]
        filename = f'curva_{nombre_grb}_{detector}.png'
        fig.tight_layout()
        fig.savefig(filename, dpi=150)
        plt.close(fig)
        lcplots.append(lcplot)

    for i, (data_spec, bkgd_spec, src_spec) in enumerate(zip(data_specs, bkgd_specs, src_specs)):
        fig = plt.figure()
        specplot = Spectrum(data=data_spec, background=bkgd_spec, figure=fig)
        specplot.add_selection(src_spec)

        detector = cspecs.detector()[i]
        filename = f'espectro_{nombre_grb}_{detector}.png'
        fig.tight_layout()
        fig.savefig(filename, dpi=150)
        plt.close(fig)
        specplots.append(specplot)

    return lcplots, specplots

def plot_curvas_y_espectros(cspecs, bkgds, nombre_grb, data="GRBs_data.dat",
                            erange_nai=(8.0, 900.0), erange_bgo=(650, 35000.0)):
    """Genera los objetos de curvas de luz y espectros, con sus selecciones para el rango fuente."""
    T_start, T_90, *_ = cargar_configuracion_grb(nombre_grb, data)
    view_range = (T_start - 5.0, T_start + T_90 + 10.0)
    src_range = (T_start, T_start + T_90)
    
    # Conversión a curvas de luz
    data_lcs = cspecs.to_lightcurve(nai_kwargs={'energy_range': erange_nai}, bgo_kwargs={'energy_range': erange_bgo})
    bkgd_lcs = bkgds.integrate_energy(nai_args=erange_nai, bgo_args=erange_bgo)
    src_lcs = cspecs.to_lightcurve(time_range=src_range, nai_kwargs={'energy_range': erange_nai}, bgo_kwargs={'energy_range': erange_bgo})

    # Conversión a espectros
    data_specs = cspecs.to_spectrum(time_range=src_range)
    bkgd_specs = bkgds.integrate_time(*src_range)
    src_specs = cspecs.to_spectrum(time_range=src_range, nai_kwargs={'energy_range': erange_nai}, bgo_kwargs={'energy_range': erange_bgo})

    # Plot de curvas de luz
    lcplots = [Lightcurve(data=data_lc, background=bkgd_lc) for data_lc, bkgd_lc in zip(data_lcs, bkgd_lcs)]
    _ = [lcplot.add_selection(src_lc) for lcplot, src_lc in zip(lcplots, src_lcs)]

    for lcplot in lcplots:
        lcplot.xlim = view_range
        plt.tight_layout()
        plt.savefig(f'curva_{nombre_grb}.png', dpi=150)

    # Plot de espectros
    specplots = [Spectrum(data=data_spec, background=bkgd_spec) for data_spec, bkgd_spec in zip(data_specs, bkgd_specs)]
    _ = [specplot.add_selection(src_spec) for specplot, src_spec in zip(specplots, src_specs)]

    return lcplots, specplots

def percent_error(value, error):
    return (error / value) * 100

def comptonized_flux_ergs(A_compton, E_peak, E_piv=100, photon_index=-0.7, E_min=10.0, E_max=1000.0):
    """ Calcula el flujo en unidades de erg/s/cm^2 para una función Comptonized """
    def flux_integrand(E):
        return A_compton * np.exp(-(2 + photon_index) * (E / E_peak)) * (E / E_piv) ** photon_index
    
    flux_ph, error_ph = quad(flux_integrand, E_min, E_max)
    
    energia_prom = (E_max + E_min) / 2 
    flux_ergs = flux_ph * energia_prom * 1.602e-9
    error_ergs = error_ph * energia_prom * 1.602e-9  # Propagación del error de la integral

    return flux_ergs, error_ergs


def ajustemulticomp_grb(nombre_grb, cspecs, bkgds, rsps, bin_inicial=0.512, bin_max=4,
                        erange_nai=(8.0, 900.0), erange_bgo=(650, 35000.0), margen_error=10,
                        data="GRBs_data.dat"):

    T_start, T_90, *_ = cargar_configuracion_grb(nombre_grb, data)

    archivo_salida = f"{nombre_grb}_ajuste.dat"

    tiempo_inicial = T_start
    tiempo_final = T_start + T_90
    limite_sup = tiempo_final

    mejores_intervalos = []

    with open(archivo_salida, 'w') as file:
        file.write('Tiempo_i(s) Tiempo_f(s) Epeak(keV) -delta_Epeak +delta_Epeak kT(keV) -delta1_kT +delta2_kT Acomp -delta_Acomp +delta_Acomp Cstat/DoF Flujo(erg/s/cm^2) -delta_Flujo +delta_Flujo\n')

        while tiempo_inicial + bin_inicial <= tiempo_final:
            bin_actual = bin_inicial
            bin_encontrado = False

            while bin_actual <= bin_max and not bin_encontrado:
                rango_actual = (tiempo_inicial, tiempo_inicial + bin_actual)

                if rango_actual[1] > limite_sup:
                    print(f"Deteniendo búsqueda, límite superior alcanzado: {rango_actual[1]}")
                    break

                print(f"Probando intervalo: {rango_actual[0]} - {rango_actual[1]}")

                phas = cspecs.to_pha(time_ranges=rango_actual, nai_kwargs={'energy_range': erange_nai}, bgo_kwargs={'energy_range': erange_bgo})
                rsps_interp = [rsp.interpolate(pha.tcent) for rsp, pha in zip(rsps, phas)]

                specfitter = SpectralFitterCstat(phas, bkgds.to_list(), rsps.to_list(), method='TNC')
                c_bb_pl = FixedComptonized() + BlackBody() + FixedPowerLaw()
                specfitter.fit(c_bb_pl, options={'maxiter': 1000})

                A_compton = specfitter.parameters[0]
                e_peak_value = specfitter.parameters[1]
                e_peak_error = specfitter.asymmetric_errors(cl=0.9)[1]
                kT_value = specfitter.parameters[3]
                kT_error = specfitter.asymmetric_errors(cl=0.9)[3]
                A_err = specfitter.asymmetric_errors(cl=0.9)[0]

                max_error = np.max(np.abs(e_peak_error))
                error_percent = percent_error(e_peak_value, max_error)
                print(f"Error porcentaje para E_peak: {error_percent:.2f}%")

                if error_percent <= margen_error:
                    flux_ergs, error_flux = comptonized_flux_ergs(A_compton, e_peak_value)
                    error_flux_total = np.sqrt((error_flux)**2 + ((A_err[1] / A_compton) * flux_ergs)**2 + ((max_error / e_peak_value) * flux_ergs)**2)

                    cstat_dof = f'{specfitter.statistic}/{specfitter.dof}'
                    file.write(f'{rango_actual[0]} {rango_actual[1]} {e_peak_value} {e_peak_error[0]} {e_peak_error[1]} {kT_value} {kT_error[0]} {kT_error[1]} {A_compton} {A_err[0]} {A_err[1]} {cstat_dof} {flux_ergs:.6e} {-error_flux_total:.6e} {+error_flux_total:.6e}\n')

                    mejores_intervalos.append((rango_actual[0], rango_actual[1], e_peak_value, max_error, kT_value, kT_error, A_compton, A_err, cstat_dof, flux_ergs, error_flux_total))
                    bin_encontrado = True
                    print(f"Se encontró el rango de tiempo óptimo: {rango_actual[0]} - {rango_actual[1]} con error en E_peak dentro del {margen_error}%, flujo {flux_ergs:.2e} erg/s/cm^2")

                bin_actual += bin_inicial

            if bin_encontrado:
                tiempo_inicial += bin_actual - bin_inicial
            else:
                tiempo_inicial += bin_inicial

            if tiempo_inicial >= limite_sup:
                print("Límite superior alcanzado, terminando ejecución.")
                break

    if not mejores_intervalos:
        print("No se encontraron intervalos de tiempo válidos dentro del margen de error.")
    else:
        print("Mejores intervalos encontrados:", mejores_intervalos)

def comptonized_N_E(E, A_compton, E_peak, photon_index=-0.7, E_piv=100):
    E0 = E_peak / (2 + photon_index)
    return A_compton * (E / E_piv)**photon_index * np.exp(-(2 + photon_index) * (E / E_peak))

def cargar_ajustes_y_N_E(nombre_grb, photon_index=-0.7, E_piv=100, orden=1):
    archivo_ajuste = f"{nombre_grb}_ajuste.dat"
    df = pd.read_csv(archivo_ajuste, delim_whitespace=True)

    funciones_N_E = []

    for _, row in df.iterrows():
        A_compton = row['Acomp']
        E_peak = row['Epeak(keV)']

        def N_E(E, A=A_compton, Ep=E_peak, alpha=photon_index, Epiv=E_piv):
            return comptonized_N_E(E, A, Ep, alpha, Epiv)

        funciones_N_E.append(N_E)

    return df, funciones_N_E

def buscar_z_optimo3(nombre_grb, photon_index=-0.7, E_piv=100, Emin=8.0, Emax=1000.0, E1=1.0, E2=10000.0,
                     z_min=0.01, z_max=10.0, z_steps=200, A_target=9.6e52, plot=True):
    """Busca el z que mejor ajusta la amplitud de la correlación L-Epeak_rest considerando errores."""
    df, funciones_N_E = cargar_ajustes_y_N_E(nombre_grb, photon_index, E_piv)

    best_z = None
    best_slope = None
    best_Es = None
    best_Ls = None
    best_dEmin = None
    best_dEplus = None
    best_Lerr = None
    min_diff = np.inf

    #zs = np.logspace(np.log10(z_min), np.log10(z_max), z_steps)
    zs = np.arange(z_min, z_max, 0.0002)
    diffs = []

    for z in zs:
        kc_values = []
        Lp_values = []
        Lp_errors = []
        Epeak_rest_values = []
        dE_min = []
        dE_plus = []

        for idx, N_E in enumerate(funciones_N_E):
            numerator, _ = quad(lambda E: E * N_E(E * (1 + z)), Emin, Emax)
            denominator, _ = quad(lambda E: E * N_E(E), Emin, Emax)
            kc = numerator / denominator if denominator != 0 else np.nan

            if np.isnan(kc):
                continue

            F_gamma = df.iloc[idx]['Flujo(erg/s/cm^2)']
            cosmo = FlatLambdaCDM(H0=67.36, Om0=0.3166, Tcmb0=2.725)
            D_L = cosmo.luminosity_distance(z).cgs.value  # en cm
            L_p = 4 * np.pi * D_L**2 * F_gamma * kc
            Lp_err = 0.1 * L_p  # 10% de error en Lp

            E_peak_obs = df.iloc[idx]['Epeak(keV)']
            E_peak_rest = E_peak_obs * (1 + z)
            e_err_minus = df.iloc[idx]['-delta_Epeak'] * (1 + z)
            e_err_plus = df.iloc[idx]['+delta_Epeak'] * (1 + z)

            kc_values.append(kc)
            Lp_values.append(L_p)
            Lp_errors.append(Lp_err)
            Epeak_rest_values.append(E_peak_rest)
            dE_min.append(e_err_minus)
            dE_plus.append(e_err_plus)

        if len(Lp_values) < 2:
            continue

        #print(f"z = {z:.3f} | len(E) = {len(Epeak_rest_values)}, len(L) = {len(Lp_values)}")

        logE = np.log10(Epeak_rest_values)
        logL = np.log10(Lp_values)

        # Propagar error a logE y logL (log(1 ± Δx/x)) ≈ Δx/x / ln(10)
        logL_err = np.array([0.1 / np.log(10)] * len(logL))
        logE_err = np.array([0.5 * (dE_plus[i] + dE_min[i]) / (Epeak_rest_values[i] * np.log(10)) for i in range(len(Epeak_rest_values))])

        # Combinar errores cuadráticamente si se quiere usar en ajuste ponderado
        total_log_err = np.sqrt(logL_err**2 + logE_err**2)

        def linear_log(x, a, b):
            return a * x + b

        try:
            popt, pcov = curve_fit(linear_log, logE, logL, sigma=total_log_err, absolute_sigma=True)
            slope, intercept = popt
            #diff = np.abs(10**intercept - A_target)
            diff = np.abs(intercept - np.log10(A_target))
            print(f"z={z:.6f} | A ajustada = {10**intercept:.2e} | diferencia = {diff:.6e}")
        except RuntimeError:
            continue

        diffs.append(diff)

        if diff < min_diff:
            min_diff = diff
            best_z = z
            best_slope = slope
            best_Es = Epeak_rest_values.copy()
            best_Ls = Lp_values.copy()
            best_Lerr = Lp_errors.copy()
            best_dEmin = dE_min.copy()
            best_dEplus = dE_plus.copy()
            best_amplitud = 10**intercept
    
    if plot:
        plt.figure(figsize=(8,6))
        plt.semilogx(zs[:len(diffs)], diffs, label='|Amplitud - A_target|')
        plt.xlabel('z')
        plt.ylabel('Diferencia en amplitud')
        plt.title('Optimización del redshift')
        plt.grid(True)
        plt.legend()
        
        from matplotlib.ticker import ScalarFormatter
        ax = plt.gca()
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        plt.show()

    salida = pd.DataFrame({
        'Epeak_rest(keV)': best_Es,
        '-delta_Epeak_rest': best_dEmin,
        '+delta_Epeak_rest': best_dEplus,
        'L_p(erg/s)': best_Ls,
        '-delta_L_p': [0.1 * val for val in best_Ls],
        '+delta_L_p': [0.1 * val for val in best_Ls]
    })
    archivo_salida = f"{nombre_grb}_L_Erest_bestz.dat"
    salida.to_csv(archivo_salida, sep=' ', index=False)
    print(f"Datos guardados en: {archivo_salida}")
    print(f"Mejor z encontrado: {best_z:.6f}")
    print(f"Pendiente del ajuste: {best_slope:.4f}")
    print(f"Amplitud del modelo (A): {best_amplitud:.3e}")

    return best_z, best_slope, best_amplitud

def graficar_Lp_vs_Epeak(nombre_grb, z_optimo):
    """Grafica log(Lp) vs log(Epeak_rest) con barras de error usando el archivo de salida generado y considerando pesos y errores en logE y logL."""
    archivo_salida = f"{nombre_grb}_L_Erest_bestz.dat"
    df = pd.read_csv(archivo_salida, sep='\s+')

    E = df['Epeak_rest(keV)']
    L = df['L_p(erg/s)']
    E_err_min = df['-delta_Epeak_rest']
    E_err_plus = df['+delta_Epeak_rest']
    L_err = df['+delta_L_p']

    logE = np.log10(E)
    logL = np.log10(L)

    logL_err = L_err / (L * np.log(10))
    logE_err = 0.5 * (E_err_min + E_err_plus) / (E * np.log(10))
    total_log_err = np.sqrt(logL_err**2 + logE_err**2)

    def linear_log(x, a, b):
        return a * x + b

    popt, _ = curve_fit(linear_log, logE, logL, sigma=total_log_err, absolute_sigma=True)
    slope, intercept = popt
    fit_y = 10**(linear_log(logE, *popt))

    plt.figure(figsize=(8,6))
    plt.errorbar(E, L, xerr=[E_err_min, E_err_plus], yerr=L_err, fmt='o', ecolor='gray', capsize=3, label='Datos')
    plt.plot(E, fit_y, color='red', label=f'Ajuste ponderado (pendiente = {slope:.2f})')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$E_{\text{peak}}^{\text{rest}}$ [keV]')
    plt.ylabel(r'$L_{\text{p}}$ [erg/s]')
    plt.title(f'Correlación $L_p$ vs $E_{{\text{{peak,rest}}}}$ para {nombre_grb} (z={z_optimo:.3f})')
    plt.grid(True, which='both', ls='--')
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"Pendiente del ajuste ponderado: {slope:.4f}")
    