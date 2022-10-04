import math
import fourier
import numpy

# This test script tests for "close enough" equality. Your results may differ
# slightly from mine due to round off error in each of our procedures.

simpsons_rule_test_1 = 0.3333333333333334

simpsons_rule_1 = fourier.simpsons_rule(lambda x : x**2, 0, 1, 100)

if numpy.allclose(simpsons_rule_1, simpsons_rule_test_1, 
                  rtol = 1e-6, atol = 1e-10):
    print("Success! Simpson's Rule test 1 passed.")
else:
    print("Failure...Simpson's Rule test 1 failed.")






simpsons_rule_test_2 = 1.5707963267948948

simpsons_rule_2 = fourier.simpsons_rule(lambda x : math.cos(x)**2, 0,
                                        math.pi, 10000)

if numpy.allclose(simpsons_rule_2, simpsons_rule_test_2,
                  rtol = 1e-6, atol = 1e-10):
    print("Success! Simpson's Rule test 2 passed.")
else:
    print("Failure...Simpson's Rule test 2 failed.")






simpsons_rule_test_3 = 0.03147144694246465

simpsons_rule_3 = fourier.simpsons_rule(lambda x: math.exp(-x), 3, 4, 10)

if numpy.allclose(simpsons_rule_3, simpsons_rule_test_3,
                  rtol = 1e-6, atol = 1e-10):
    print("Success! Simpson's Rule test 3 passed.")
else:
    print("Failure...Simpson's Rule test 3 failed.")
    
    
    
    
    
        
def f(x):
    return x**2 + math.cos(x)

fc_test_1 = (6.579736267392905, 
             (-2.999999999999983, 0.9999999999999555, -0.44444444444434916,
              0.24999999999983422, -0.15999999999974093, 0.11111111111073693,
              -0.08163265306071445, 0.06249999999933395, -0.04938271604853894,
              0.03999999999896045),
             (-2.810090123558003e-17, 5.3139437294902336e-17,
              2.997694685073308e-17, -1.22894287339174e-16,
              2.0620357273533801e-16, -8.205325308097143e-16,
              3.60491266467496e-16, 6.373438813748559e-16,
              1.9159488810297868e-16, -1.024384281104555e-16))

fc_1 = fourier.calculate_fourier_coefs(f, 10)


if numpy.allclose(fc_1[0], fc_test_1[0]) and \
        numpy.allclose(fc_1[1], fc_test_1[1]) and \
        numpy.allclose(fc_1[2], fc_test_1[2]):
    print("Success! Fourier coefficient test 1 passed.")
else:
    print("Failure... Fourier coefficient test 1 failed.")



    
    

def g(x):
    return x * math.exp(-x)

fc_test_2 = (-15.831750730293086,
             (11.591953275521588, -5.5190400086983, 2.906563120763606,
              -1.7453589919675692, 1.1527119970766324, -0.8145580687443159,
              0.6048395227754075, -0.46630526449792875, 0.37020416460478045,
              -0.3008957933368353, 0.24930766578708163, -0.20989416085751134),
             (7.915875365146437, -8.097217689097207, 6.514042616068149,
              -5.251516951229, 4.349683866019909, -3.6951069280406847,
              3.2045748445560247, -2.825561399633252, 2.5248935499685814,
              -2.2810217135901723, 2.079485028475204, -1.9102756556395624))

fc_2 = fourier.calculate_fourier_coefs(g, 12)

if numpy.allclose(fc_2[0], fc_test_2[0]) and \
        numpy.allclose(fc_2[1], fc_test_2[1]) and \
        numpy.allclose(fc_2[2], fc_test_2[2]):
    print("Success! Fourier coefficient test 2 passed.")
else:
    print("Failure... Fourier coefficient test 2 failed.")
    
    



    
def h(x):
    return 1.0/(x**2+x+1)

fc_test_3 = (0.9524443041144453,
                 (0.4496482886116936,
                  0.10222688041433228,
                  0.010079300076073406,
                  -0.017387472453887835,
                  -0.010645625825162274,
                  -0.007408683658014924,
                  -0.0017206828985502545,
                  -0.0013535598489071996),
                 (-0.24396128241975884,
                  -0.1642713093854458,
                  -0.09130192483617189,
                  -0.028507117073205804,
                  -0.012653973922086725,
                  0.002093967044695628,
                  -0.0016427176220543149,
                  0.003129538610424603))


fc_3 = fourier.calculate_fourier_coefs(h, 8)

if numpy.allclose(fc_3[0], fc_test_3[0]) and \
        numpy.allclose(fc_3[1], fc_test_3[1]) and \
        numpy.allclose(fc_3[2], fc_test_3[2]):
    print("Success! Fourier coefficient test 3 passed.")
else:
    print("Failure... Fourier coefficient test 3 failed.")
    





x_values_1 = numpy.linspace(-math.pi, math.pi, 11)

approx_test_1 = numpy.array([8.4889390583586,
                             5.510489340628741,
                             3.2585828167423174,
                             1.9051712779844672,
                             1.2216179883162666,
                             1.0180194312519468,
                             1.2216179883162648,
                             1.905171277984465,
                             3.25858281674232,
                             5.510489340628741,
                             8.4889390583586])

    
approx_1 = fourier.calculate_fourier_approx(fc_1, x_values_1)


if numpy.allclose(approx_1, approx_test_1):
    print("Success! Fourier approximation calculation 1 passed.")
else:
    print("Failure... Fourier approximation calculation 1 failed.")






x_values_2 = numpy.linspace(-math.pi, math.pi, 21)


approx_test_2 = numpy.array([-33.84750739977812,
                             -52.21103524293375,
                             -30.985455892532762,
                             -18.35645719611176,
                             -13.982032484477557,
                             -6.6952902037256505,
                             -4.32867725642939,
                             -3.210304034823552,
                             -0.21196440651946213,
                             -1.0220499124616764,
                             -0.09634790671990773,
                             0.9576057392017319,
                             -0.6254382187689198,
                             1.0001925773683196,
                             0.5080828169133336,
                             -0.7105896196545508,
                             1.819526731743987,
                             -0.8951344073953188,
                             -0.41789756826499636,
                             4.993266582439128,
                             -33.84750739977807])


approx_2 = fourier.calculate_fourier_approx(fc_2, x_values_2)

if numpy.allclose(approx_2, approx_test_2):
    print("Success! Fourier approximation calculation 2 passed.")
else:
    print("Failure... Fourier approximation calculation 2 failed.")
    
    




x_values_3 = numpy.linspace(-math.pi, math.pi, 31)

approx_test_3 = numpy.array([0.10493803654669051,
                             0.14743910157933451,
                             0.18037907956804553,
                             0.20709173996943625,
                             0.24772274159561156,
                             0.30538451537952954,
                             0.37543863272477845,
                             0.46759355028465327,
                             0.5943312549749281,
                             0.7571055581619223,
                             0.9522378541712115,
                             1.156988805291973,
                             1.3051523625989874,
                             1.3218649030439447,
                             1.1981795981451375,
                             0.9996605964747993,
                             0.7984257762109916,
                             0.6274455489340719,
                             0.4934276616549707,
                             0.39398739074633693,
                             0.3190266142515008,
                             0.2598795644548403,
                             0.21596452245591774,
                             0.18384255401308675,
                             0.15538219588649768,
                             0.1316331935715143,
                             0.11737346316036064,
                             0.1041451720993036,
                             0.08535248691271045,
                             0.07927008685359234,
                             0.10493803654669046])

approx_3 = fourier.calculate_fourier_approx(fc_3, x_values_3)

if numpy.allclose(approx_3, approx_test_3):
    print("Success! Fourier approximation calculation 3 passed.")
else:
    print("Failure... Fourier approximation calculation 3 failed.")