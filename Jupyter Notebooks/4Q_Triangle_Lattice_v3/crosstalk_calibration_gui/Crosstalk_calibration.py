import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.io import loadmat
from matplotlib import rcParams
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
rcParams['figure.figsize'] = 12, 5
plt.rcParams.update({'font.size': 15})
plotsize = (10, 6)
legend_size = 12
rcParams["savefig.dpi"] = 300
rcParams["savefig.bbox"] = 'tight'

def get_voltage_data(filename):
    '''
    Gets cwvector (locations of transmission minima) vs Voltage data from a .mat file
    args:
        filename: name of .mat file
    returns:
        V: Voltage data (Volts)
        cw_Freq: Frequency point of minimum amplitude of transmission (in GHz)
        name: Name of object being swept
    '''
    mat = loadmat(filename)
    try:
        voltages = mat['random_voltages_matrix']
    except:
        voltages = mat['voltage_matrix']

    return voltages

def Zfile(label):
    return r'Z://QSimMeasurements/Measurements/4Q_Triangle_Lattice/pnax{}24/2Tone4Qubit_NR_2024{}.mat'.format(label[:4], label)


def errorfn(crosstalk_row, voltages, fluxes, flux_quanta, offset_value, index, value_to_add = 1, fit_offset_vector = False):
    crosstalk_row = np.insert(crosstalk_row, index, value_to_add)

    if fit_offset_vector:
        offset_value = crosstalk_row[-1]
        crosstalk_row = np.delete(crosstalk_row, -1)
    expected_fluxes = []
    for i, v in enumerate(voltages):
        flux_value = crosstalk_row.dot(np.array(v))
        expected_fluxes.append((flux_value - offset_value) / flux_quanta)
    error_value = np.sum(( np.array(expected_fluxes) - np.array(fluxes)) ** 2)
    print(error_value)
    return(error_value)

def estimate_xc(fluxes, voltages, crosstalk_guess, offset_value, flux_quanta, index, fit_offset_vector = False):
    res = minimize(lambda x: errorfn(x, voltages, fluxes, flux_quanta, offset_value, index,
                                     fit_offset_vector=fit_offset_vector), crosstalk_guess,
                   method='Nelder-Mead', tol = 0.000001)
    return res

def ignore_certain_indeces(list_of_ignored_vals, fluxes, voltages):
    new_voltages = []
    new_fluxes = []
    for i in range(len(fluxes)):
        if i in list_of_ignored_vals:
            continue
        new_voltages.append(voltages[i])
        new_fluxes.append(fluxes[i])

    return(np.array(new_voltages), np.array(new_fluxes))

qubit_number = 12

if qubit_number == 2:
    flux_numbers = [0.2950936392987421, 0.29519804407605627, 0.2952423265603364, 0.29506306801942056, 0.29548028547857613,
                    0.2951405083100034, 0.2951097775612734, 0.2950578683876315, 0.2951065742759179, 0.2951214281553099,
                    0.295072311634605, 0.29513666903559704, 0.29518760139616357, 0.2950790963168394, 0.29530622029436226,
                    0.29513997275003734, 0.2951061480715071, 0.2950416205171088, 0.2951680393391523, 0.2950116643677704,
                    0.2951547508899339, 0.29511740787394897, 0.2951907692489146, 0.2954675228703208, 0.2951746790198587,
                    0.2950541263997631, 0.2951705927297803, 0.29524251724783074, 0.2955003824600484, 0.2950632313136149,
                    0.2951414684766794, 0.2951746251998004, 0.2950527276467552, 0.2950525517213204, 0.2950325302821383,
                    0.29510881238715975, 0.2951703625107996, 0.2950952757291677, 0.29504511493352675, 0.2950517386131087,
                    0.2950164846142104, 0.29512429090959413, 0.2951244352533125, 0.2950989244977834, 0.29511582021480026,
                    0.2951388384471055, 0.29508187245357625, 0.29503372944517153, 0.2951139060294059, 0.2951028769367547]
    flux_numbers = [-1 * x for x in flux_numbers]
    flux_quanta = 2.805
    offset_value = 0.57766812

    Index_for_fitting = None

    indexes_to_ignore = [2]


    expected_qubit_flux = -0.295

    row_values = np.array([1, 0.00153105,  0.00163795, -0.02037039, -0.01961468, -0.00841654,  0.01806841,
              0.01414701])
    index_of_measured_value = 0
    file_name = '0717_1225'
    fit_offset_vector = True


if qubit_number == 3:
    flux_numbers = [0.31419004017142116, 0.3140712034800845, 0.3138733302740554, 0.3138620966793315, 0.3139226483386203,
                    0.3138981838660046, 0.3136523249692691, 0.31245046269037235, 0.31388798675431545, 0.3140989683053762,
                    0.31319178320629304, 0.3142784805906371, 0.3140185771758671, 0.31360832547594203, 0.31390542195884563,
                    0.31399882673715224, 0.3138540104389121, 0.3140018377601046, 0.3132045453140064, 0.3143688106353888,
                    0.31392874007136706, 0.3134176011048064, 0.31282169942902305, 0.31371492213546887, 0.3129369728846981]
    flux_quanta = 2.714
    offset_value = -0.617979

    Index_for_fitting = None

    expected_qubit_flux = 0.314
    indexes_to_ignore = []


    row_values = np.array([-0.0003331671351100985, 1, 0.003470467606387177, -0.01198468781737309, -0.026560659486571877,
                           -0.054416960899318814, 0.027208480449658977, 0.01496466467697475])
    index_of_measured_value = 1
    file_name = '0716_1622'
    fit_offset_vector = True

if qubit_number == 4:
    flux_numbers = [0.31621406074523317, 0.31554025385707307, 0.3165659148871745, 0.31659658463916385, 0.31729102641591833,
                    0.31695494861539475, 0.31511917216215457, 0.31636317177007733, 0.31622748916207083, 0.3160765728484086,
                    0.3161682716135082, 0.3173405660790704, 0.3150745964593725, 0.3164561360849621, 0.3163336444513618,
                    0.31573499104454256, 0.3172677993342137, 0.31623841859912033, 0.3165897488208132, 0.31540276604403944,
                    0.31528850380386586, 0.3150089211526958, 0.31705833555967755, 0.3164353067874472, 0.3166290901153404,
                    0.31602047252050114, 0.3171129684281068, 0.3165796750930307, 0.31675535979190894, 0.31880569597567743,
                    0.3188511192704837, 0.3166593460902114, 0.31626816797723595, 0.3169023075055071, 0.3159539931147438,
                    0.31588878269653314, 0.3160135099861831, 0.319106672617158, 0.3155587255293015, 0.31558365103545927,
                    0.315373736544093, 0.31700277773753804, 0.3169733953402129, 0.31642795340088403, 0.31786726770629214,
                    0.31455177522997974, 0.31654386301530435, 0.31518413342617013, 0.3162947572469488, 0.31634338859109057,
                    0.31552232724381474, 0.3165552975597203, 0.31651880670114546, 0.3159055485407169, 0.3163491977368283,
                    0.3160572608053524, 0.31854301706341875, 0.31627427134639113, 0.3165087820161943, 0.31457660016796746,
                    0.31624770974636, 0.31636811902177664, 0.3158506440045003, 0.3174070115107057, 0.31519854886486165,
                    0.3164397861355315, 0.31622672354388426, 0.31672852407544616, 0.3175520627463222]

    row_values = [-0.02437037014213388, 0.028000000233000633, 1, -0.013703703663201715, 0.020740740679690128, 0.030370370280974546, 0.038095238274805335, 0.047089946951535094],


    flux_numbers = [-1 * x for x in flux_numbers]
    flux_quanta = 2.713
    offset_value = 0.5105258

    expected_qubit_flux = -0.317

    indexes_to_ignore = [47, 59]

    Index_for_fitting = 35

    index_of_measured_value = 2
    file_name = '0716_1951'
    fit_offset_vector = True

if qubit_number == 12:
    flux_numbers = [-0.019156104782836827, -0.0023946561007076967,
                         0,
                         -0.029766496815472526,
                         0,
                         -0.01804374154034447,
                         -0.04072909848246882,
                         -0.010235483569518861,
                         -0.016733513428028344,
                         -0.011818774467389081,
                         0,
                         0,
                         -0.015464343475944952,
                         -0.006340156775510266,
                         -0.008363475704284404,
                         0,
                         -0.0180353868472911,
                         -0.014285885867173592,
                         -0.012699148794740885,
                         -0.004041479823168596,
                         0,
                         -0.0791822896844603,
                         -0.01646173380955864,
                         -0.014897144677457954,
                         -0.0033292409815674265,
                         -0.02685286471741667,
                         -0.008463420427582913,
                         -0.010711722472842742]
    flux_quanta = 2.11561703
    offset_value = -0.18158 - 0.0134

    expected_qubit_flux = -0.01

    indexes_to_ignore = [2, 4, 10, 11, 15, 20]

    Index_for_fitting = None

    row_values = np.array([-0.3550834171462236, 0.39434280759997997, -0.2930595767885369, 1, 0.9180246838750391,
                           0.42189816389697987, 0.09866429511668051, 0.18237689194134155]
)
    index_of_measured_value = 3
    file_name = '0717_1507'
    fit_offset_vector = True



voltages = get_voltage_data(Zfile(file_name))

print(voltages)

voltages, flux_numbers = ignore_certain_indeces(indexes_to_ignore, flux_numbers, voltages)

row_values = np.delete(row_values, index_of_measured_value)

if fit_offset_vector:
    row_values = np.insert(row_values, len(row_values), offset_value)


result = estimate_xc(flux_numbers[:Index_for_fitting], voltages[:Index_for_fitting], row_values, offset_value, flux_quanta, index_of_measured_value,
                     fit_offset_vector = fit_offset_vector)
print('Final x', result['x'])

new_vector = result['x']
if fit_offset_vector:
    fitted_offset_vector = new_vector[-1]
    new_vector = np.delete(new_vector, -1)
new_vector = np.insert(new_vector, index_of_measured_value, 1)

fitted_fluxes = []
for i, v in enumerate(voltages):
    flux_value = new_vector.dot(np.array(v))
    if fit_offset_vector:
        fit_offset = fitted_offset_vector
    else:
        fit_offset = offset_value
    fitted_fluxes.append((flux_value - fit_offset) / flux_quanta)


difference = np.array(flux_numbers) - np.array(fitted_fluxes)
difference_original = np.array(flux_numbers) - np.array(expected_qubit_flux)

print('Mean abs: ', np.mean(np.abs(difference)))
print('Range: ', np.max(difference) - np.min(difference))

plt.plot(difference, '.-', label = 'Optimized')
plt.plot(difference_original, '.-', label = 'Original')

plt.xlabel('Random Yoko Voltage Instance')
plt.ylabel('Qubit Number')
plt.locator_params(axis='y', nbins=4)
plt.legend()
plt.show()
