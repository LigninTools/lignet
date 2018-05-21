#!/usr/bin/env python3

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-C", "--PLIGC", type=float,
                    help="mol number of C in input", required=True)
parser.add_argument("-H", "--PLIGH", type=float,
                    help="mol number of H in input", required=True)
parser.add_argument("-O", "--PLIGO", type=float,
                    help="mol number of O in input", required=True)
parser.add_argument("-t", "--T0", type=float, default=298.15,
                    help="starting temperature", required=False)
parser.add_argument("-m", "--max_temperature", type=float, 
                    help="maximum pyrolysis temperature", default=float('nan'))
parser.add_argument("-r", "--heating_rate", type=float, default=2.7,
                    help="heating rate in K/s")
parser.add_argument("-s", "--end_time", type=float, default=600,
                    help="end_time in seconds")
args = parser.parse_args()

print("Solver arguments are:")
print("\tPLIGC\t%f" % args.PLIGC)
print("\tPLIGH\t%f" % args.PLIGH)
print("\tPLIGO\t%f" % args.PLIGO)
print("\ttemperature\t%f" % args.T0)
print("\tmax temperature\t%f" % args.max_temperature)
print("\theating rate\t%f" % args.heating_rate)
print("\tend time (s)\t%f\n" % args.end_time)

T0 = args.T0
PLIGC = args.PLIGC
PLIGH = args.PLIGH
PLIGO = args.PLIGO
alpha = args.heating_rate
T_max = args.max_temperature
end_time = args.end_time
stoptime = min((args.max_temperature-args.T0)/args.heating_rate,600)

from scipy.integrate import odeint
import time
import numpy as np


def ODEs(y, t, p):
	T, ADIO, ADIOM2, ALD3, C10H2, C10H2M2, C10H2M4, C2H6, C3H4O, C3H4O2, C3H6, C3H6O2, C3H8O2, CH2CO, CH3CHO, CH3OH, CH4, CHAR, CO, CO2, COUMARYL, ETOH, H2, H2O, KET, KETD, KETDM2, KETM2, LIG, LIGC, LIGH, LIGM2, LIGO, MGUAI, OH, PADIO, PADIOM2, PC2H2, PCH2OH, PCH2P, PCH3, PCHO, PCHOHP, PCHP2, PCOH, PCOHP2, PCOS, PFET3, PFET3M2, PH2, PHENOL, PKETM2, PLIG, PLIGC, PLIGH, PLIGM2, PLIGO, PRADIO, PRADIOM2, PRFET3, PRFET3M2, PRKETM2, PRLIGH, PRLIGH2, PRLIGM2A, RADIO, RADIOM2, RC3H3O, RC3H5O2, RC3H7O2, RCH3, RCH3O, RKET, RKETM2, RLIGA, RLIGB, RLIGH, RLIGM2A, RLIGM2B, RMGUAI, RPHENOL, RPHENOX, RPHENOXM2, SYNAPYL, VADIO, VADIOM2, VCOUMARYL, VKET, VKETD, VKETDM2, VKETM2, VMGUAI, VPHENOL, VSYNAPYL = y
	alpha, R, A0, n0, E0, A1, n1, E1, A2, n2, E2, A3, n3, E3, A4, n4, E4, A5, n5, E5, A6, n6, E6, A7, n7, E7, A8, n8, E8, A9, n9, E9, A10, n10, E10, A11, n11, E11, A12, n12, E12, A13, n13, E13, A14, n14, E14, A15, n15, E15, A16, n16, E16, A17, n17, E17, A18, n18, E18, A19, n19, E19, A20, n20, E20, A21, n21, E21, A22, n22, E22, A23, n23, E23, A24, n24, E24, A25, n25, E25, A26, n26, E26, A27, n27, E27, A28, n28, E28, A29, n29, E29, A30, n30, E30, A31, n31, E31, A32, n32, E32, A33, n33, E33, A34, n34, E34, A35, n35, E35, A36, n36, E36, A37, n37, E37, A38, n38, E38, A39, n39, E39, A40, n40, E40, A41, n41, E41, A42, n42, E42, A43, n43, E43, A44, n44, E44, A45, n45, E45, A46, n46, E46, A47, n47, E47, A48, n48, E48, A49, n49, E49, A50, n50, E50, A51, n51, E51, A52, n52, E52, A53, n53, E53, A54, n54, E54, A55, n55, E55, A56, n56, E56, A57, n57, E57, A58, n58, E58, A59, n59, E59, A60, n60, E60, A61, n61, E61, A62, n62, E62, A63, n63, E63, A64, n64, E64, A65, n65, E65, A66, n66, E66, A67, n67, E67, A68, n68, E68, A69, n69, E69, A70, n70, E70, A71, n71, E71, A72, n72, E72, A73, n73, E73, A74, n74, E74, A75, n75, E75, A76, n76, E76, A77, n77, E77, A78, n78, E78, A79, n79, E79, A80, n80, E80, A81, n81, E81, A82, n82, E82, A83, n83, E83, A84, n84, E84, A85, n85, E85, A86, n86, E86, A87, n87, E87, A88, n88, E88, A89, n89, E89, A90, n90, E90, A91, n91, E91, A92, n92, E92, A93, n93, E93, A94, n94, E94, A95, n95, E95, A96, n96, E96, A97, n97, E97, A98, n98, E98, A99, n99, E99, A100, n100, E100, A101, n101, E101, A102, n102, E102, A103, n103, E103, A104, n104, E104, A105, n105, E105, A106, n106, E106, A107, n107, E107, A108, n108, E108, A109, n109, E109, A110, n110, E110, A111, n111, E111, A112, n112, E112, A113, n113, E113, A114, n114, E114, A115, n115, E115, A116, n116, E116, A117, n117, E117, A118, n118, E118, A119, n119, E119, A120, n120, E120, A121, n121, E121, A122, n122, E122, A123, n123, E123, A124, n124, E124, A125, n125, E125, A126, n126, E126, A127, n127, E127, A128, n128, E128, A129, n129, E129, A130, n130, E130, A131, n131, E131, A132, n132, E132, A133, n133, E133, A134, n134, E134, A135, n135, E135, A136, n136, E136, A137, n137, E137, A138, n138, E138, A139, n139, E139, A140, n140, E140, A141, n141, E141, A142, n142, E142, A143, n143, E143, A144, n144, E144, A145, n145, E145, A146, n146, E146, A147, n147, E147, A148, n148, E148, A149, n149, E149, A150, n150, E150, A151, n151, E151, A152, n152, E152, A153, n153, E153, A154, n154, E154, A155, n155, E155, A156, n156, E156, A157, n157, E157, A158, n158, E158, A159, n159, E159, A160, n160, E160, A161, n161, E161, A162, n162, E162, A163, n163, E163, A164, n164, E164, A165, n165, E165, A166, n166, E166, A167, n167, E167, A168, n168, E168, A169, n169, E169, A170, n170, E170, A171, n171, E171, A172, n172, E172, A173, n173, E173, A174, n174, E174, A175, n175, E175, A176, n176, E176, A177, n177, E177, A178, n178, E178, A179, n179, E179, A180, n180, E180, A181, n181, E181, A182, n182, E182, A183, n183, E183, A184, n184, E184, A185, n185, E185, A186, n186, E186, A187, n187, E187, A188, n188, E188, A189, n189, E189, A190, n190, E190, A191, n191, E191, A192, n192, E192, A193, n193, E193, A194, n194, E194, A195, n195, E195, A196, n196, E196, A197, n197, E197, A198, n198, E198, A199, n199, E199, A200, n200, E200, A201, n201, E201, A202, n202, E202, A203, n203, E203, A204, n204, E204, A205, n205, E205, A206, n206, E206, A207, n207, E207, A208, n208, E208, A209, n209, E209, A210, n210, E210, A211, n211, E211, A212, n212, E212, A213, n213, E213, A214, n214, E214, A215, n215, E215, A216, n216, E216, A217, n217, E217, A218, n218, E218, A219, n219, E219, A220, n220, E220, A221, n221, E221, A222, n222, E222, A223, n223, E223, A224, n224, E224, A225, n225, E225, A226, n226, E226, A227, n227, E227, A228, n228, E228, A229, n229, E229, A230, n230, E230, A231, n231, E231, A232, n232, E232, A233, n233, E233, A234, n234, E234, A235, n235, E235, A236, n236, E236, A237, n237, E237, A238, n238, E238, A239, n239, E239, A240, n240, E240, A241, n241, E241, A242, n242, E242, A243, n243, E243, A244, n244, E244, A245, n245, E245, A246, n246, E246, A247, n247, E247, A248, n248, E248, A249, n249, E249, A250, n250, E250, A251, n251, E251, A252, n252, E252, A253, n253, E253, A254, n254, E254, A255, n255, E255, A256, n256, E256, A257, n257, E257, A258, n258, E258, A259, n259, E259, A260, n260, E260, A261, n261, E261, A262, n262, E262, A263, n263, E263, A264, n264, E264, A265, n265, E265, A266, n266, E266, A267, n267, E267, A268, n268, E268, A269, n269, E269, A270, n270, E270, A271, n271, E271, A272, n272, E272, A273, n273, E273, A274, n274, E274, A275, n275, E275, A276, n276, E276, A277, n277, E277, A278, n278, E278, A279, n279, E279, A280, n280, E280, A281, n281, E281, A282, n282, E282, A283, n283, E283, A284, n284, E284, A285, n285, E285, A286, n286, E286, A287, n287, E287, A288, n288, E288, A289, n289, E289, A290, n290, E290, A291, n291, E291, A292, n292, E292, A293, n293, E293, A294, n294, E294, A295, n295, E295, A296, n296, E296, A297, n297, E297, A298, n298, E298, A299, n299, E299, A300, n300, E300, A301, n301, E301, A302, n302, E302, A303, n303, E303, A304, n304, E304, A305, n305, E305, A306, n306, E306, A307, n307, E307, A308, n308, E308, A309, n309, E309, A310, n310, E310, A311, n311, E311, A312, n312, E312, A313, n313, E313, A314, n314, E314, A315, n315, E315, A316, n316, E316, A317, n317, E317, A318, n318, E318, A319, n319, E319, A320, n320, E320, A321, n321, E321, A322, n322, E322, A323, n323, E323, A324, n324, E324, A325, n325, E325, A326, n326, E326, A327, n327, E327, A328, n328, E328, A329, n329, E329, A330, n330, E330, A331, n331, E331, A332, n332, E332, A333, n333, E333, A334, n334, E334, A335, n335, E335, A336, n336, E336, A337, n337, E337, A338, n338, E338, A339, n339, E339, A340, n340, E340, A341, n341, E341, A342, n342, E342, A343, n343, E343, A344, n344, E344, A345, n345, E345, A346, n346, E346, A347, n347, E347, A348, n348, E348, A349, n349, E349, A350, n350, E350, A351, n351, E351, A352, n352, E352, A353, n353, E353, A354, n354, E354, A355, n355, E355, A356, n356, E356, A357, n357, E357, A358, n358, E358, A359, n359, E359, A360, n360, E360, A361, n361, E361, A362, n362, E362, A363, n363, E363, A364, n364, E364, A365, n365, E365, A366, n366, E366, A367, n367, E367, A368, n368, E368, A369, n369, E369, A370, n370, E370, A371, n371, E371, A372, n372, E372, A373, n373, E373, A374, n374, E374, A375, n375, E375, A376, n376, E376, A377, n377, E377, A378, n378, E378, A379, n379, E379, A380, n380, E380, A381, n381, E381, A382, n382, E382, A383, n383, E383, A384, n384, E384, A385, n385, E385, A386, n386, E386, A387, n387, E387, A388, n388, E388, A389, n389, E389, A390, n390, E390, A391, n391, E391, A392, n392, E392, A393, n393, E393, A394, n394, E394, A395, n395, E395, A396, n396, E396, A397, n397, E397, A398, n398, E398, A399, n399, E399, A400, n400, E400, A401, n401, E401, A402, n402, E402, A403, n403, E403, A404, n404, E404, A405, n405, E405 = p
	dydt = [alpha, 
			-1*A40 * T**n40 * np.exp(-1*E40/R/T) * ADIO**1.0 * RPHENOX**1.0  -1*A44 * T**n44 * np.exp(-1*E44/R/T) * ADIO**1.0 * RPHENOXM2**1.0  -1*A90 * T**n90 * np.exp(-1*E90/R/T) * ADIO**1.0  +1*A132 * T**n132 * np.exp(-1*E132/R/T) * RADIO**1.0 * LIGH**1.0  +1*A153 * T**n153 * np.exp(-1*E153/R/T) * RADIO**1.0 * PLIGH**1.0  +1*A174 * T**n174 * np.exp(-1*E174/R/T) * RADIO**1.0 * PLIGM2**1.0  +1*A195 * T**n195 * np.exp(-1*E195/R/T) * RADIO**1.0 * LIGM2**1.0  +1*A216 * T**n216 * np.exp(-1*E216/R/T) * RADIO**1.0 * LIGM2**1.0  +1*A237 * T**n237 * np.exp(-1*E237/R/T) * RADIO**1.0 * PFET3M2**1.0  +1*A258 * T**n258 * np.exp(-1*E258/R/T) * RADIO**1.0 * ADIOM2**1.0  +1*A279 * T**n279 * np.exp(-1*E279/R/T) * RADIO**1.0 * KETM2**1.0  +1*A300 * T**n300 * np.exp(-1*E300/R/T) * RADIO**1.0 * C10H2**1.0  +1*A321 * T**n321 * np.exp(-1*E321/R/T) * RADIO**1.0 * LIG**1.0  +1*A342 * T**n342 * np.exp(-1*E342/R/T) * RADIO**1.0 * LIG**1.0  +1*A363 * T**n363 * np.exp(-1*E363/R/T) * RADIO**1.0 * PFET3**1.0  -1*A364 * T**n364 * np.exp(-1*E364/R/T) * RC3H5O2**1.0 * ADIO**1.0  -1*A365 * T**n365 * np.exp(-1*E365/R/T) * PRFET3**1.0 * ADIO**1.0  -1*A366 * T**n366 * np.exp(-1*E366/R/T) * RC3H7O2**1.0 * ADIO**1.0  -1*A367 * T**n367 * np.exp(-1*E367/R/T) * RADIOM2**1.0 * ADIO**1.0  -1*A368 * T**n368 * np.exp(-1*E368/R/T) * PRFET3M2**1.0 * ADIO**1.0  -1*A369 * T**n369 * np.exp(-1*E369/R/T) * PRLIGH**1.0 * ADIO**1.0  -1*A370 * T**n370 * np.exp(-1*E370/R/T) * RLIGM2B**1.0 * ADIO**1.0  -1*A371 * T**n371 * np.exp(-1*E371/R/T) * RLIGM2A**1.0 * ADIO**1.0  -1*A372 * T**n372 * np.exp(-1*E372/R/T) * RCH3**1.0 * ADIO**1.0  -1*A373 * T**n373 * np.exp(-1*E373/R/T) * PRKETM2**1.0 * ADIO**1.0  -1*A374 * T**n374 * np.exp(-1*E374/R/T) * RKET**1.0 * ADIO**1.0  -1*A375 * T**n375 * np.exp(-1*E375/R/T) * PRADIO**1.0 * ADIO**1.0  -1*A376 * T**n376 * np.exp(-1*E376/R/T) * RC3H3O**1.0 * ADIO**1.0  -1*A377 * T**n377 * np.exp(-1*E377/R/T) * RLIGB**1.0 * ADIO**1.0  -1*A378 * T**n378 * np.exp(-1*E378/R/T) * RLIGA**1.0 * ADIO**1.0  -1*A379 * T**n379 * np.exp(-1*E379/R/T) * PRADIOM2**1.0 * ADIO**1.0  -1*A380 * T**n380 * np.exp(-1*E380/R/T) * RMGUAI**1.0 * ADIO**1.0  -1*A381 * T**n381 * np.exp(-1*E381/R/T) * OH**1.0 * ADIO**1.0  -1*A382 * T**n382 * np.exp(-1*E382/R/T) * RCH3O**1.0 * ADIO**1.0  -1*A383 * T**n383 * np.exp(-1*E383/R/T) * RPHENOL**1.0 * ADIO**1.0  -1*A384 * T**n384 * np.exp(-1*E384/R/T) * RADIO**1.0 * ADIO**1.0  +1*A384 * T**n384 * np.exp(-1*E384/R/T) * RADIO**1.0 * ADIO**1.0  +1*A405 * T**n405 * np.exp(-1*E405/R/T) * RADIO**1.0 * KET**1.0, 
			-1*A32 * T**n32 * np.exp(-1*E32/R/T) * ADIOM2**1.0 * RPHENOXM2**1.0  -1*A36 * T**n36 * np.exp(-1*E36/R/T) * ADIOM2**1.0 * RPHENOX**1.0  -1*A84 * T**n84 * np.exp(-1*E84/R/T) * ADIOM2**1.0  +1*A115 * T**n115 * np.exp(-1*E115/R/T) * RADIOM2**1.0 * LIGH**1.0  +1*A136 * T**n136 * np.exp(-1*E136/R/T) * RADIOM2**1.0 * PLIGH**1.0  +1*A157 * T**n157 * np.exp(-1*E157/R/T) * RADIOM2**1.0 * PLIGM2**1.0  +1*A178 * T**n178 * np.exp(-1*E178/R/T) * RADIOM2**1.0 * LIGM2**1.0  +1*A199 * T**n199 * np.exp(-1*E199/R/T) * RADIOM2**1.0 * LIGM2**1.0  +1*A220 * T**n220 * np.exp(-1*E220/R/T) * RADIOM2**1.0 * PFET3M2**1.0  -1*A238 * T**n238 * np.exp(-1*E238/R/T) * RC3H5O2**1.0 * ADIOM2**1.0  -1*A239 * T**n239 * np.exp(-1*E239/R/T) * PRFET3**1.0 * ADIOM2**1.0  -1*A240 * T**n240 * np.exp(-1*E240/R/T) * RC3H7O2**1.0 * ADIOM2**1.0  -1*A241 * T**n241 * np.exp(-1*E241/R/T) * RADIOM2**1.0 * ADIOM2**1.0  +1*A241 * T**n241 * np.exp(-1*E241/R/T) * RADIOM2**1.0 * ADIOM2**1.0  -1*A242 * T**n242 * np.exp(-1*E242/R/T) * PRFET3M2**1.0 * ADIOM2**1.0  -1*A243 * T**n243 * np.exp(-1*E243/R/T) * PRLIGH**1.0 * ADIOM2**1.0  -1*A244 * T**n244 * np.exp(-1*E244/R/T) * RLIGM2B**1.0 * ADIOM2**1.0  -1*A245 * T**n245 * np.exp(-1*E245/R/T) * RLIGM2A**1.0 * ADIOM2**1.0  -1*A246 * T**n246 * np.exp(-1*E246/R/T) * RCH3**1.0 * ADIOM2**1.0  -1*A247 * T**n247 * np.exp(-1*E247/R/T) * PRKETM2**1.0 * ADIOM2**1.0  -1*A248 * T**n248 * np.exp(-1*E248/R/T) * RKET**1.0 * ADIOM2**1.0  -1*A249 * T**n249 * np.exp(-1*E249/R/T) * PRADIO**1.0 * ADIOM2**1.0  -1*A250 * T**n250 * np.exp(-1*E250/R/T) * RC3H3O**1.0 * ADIOM2**1.0  -1*A251 * T**n251 * np.exp(-1*E251/R/T) * RLIGB**1.0 * ADIOM2**1.0  -1*A252 * T**n252 * np.exp(-1*E252/R/T) * RLIGA**1.0 * ADIOM2**1.0  -1*A253 * T**n253 * np.exp(-1*E253/R/T) * PRADIOM2**1.0 * ADIOM2**1.0  -1*A254 * T**n254 * np.exp(-1*E254/R/T) * RMGUAI**1.0 * ADIOM2**1.0  -1*A255 * T**n255 * np.exp(-1*E255/R/T) * OH**1.0 * ADIOM2**1.0  -1*A256 * T**n256 * np.exp(-1*E256/R/T) * RCH3O**1.0 * ADIOM2**1.0  -1*A257 * T**n257 * np.exp(-1*E257/R/T) * RPHENOL**1.0 * ADIOM2**1.0  -1*A258 * T**n258 * np.exp(-1*E258/R/T) * RADIO**1.0 * ADIOM2**1.0  +1*A262 * T**n262 * np.exp(-1*E262/R/T) * RADIOM2**1.0 * KETM2**1.0  +1*A283 * T**n283 * np.exp(-1*E283/R/T) * RADIOM2**1.0 * C10H2**1.0  +1*A304 * T**n304 * np.exp(-1*E304/R/T) * RADIOM2**1.0 * LIG**1.0  +1*A325 * T**n325 * np.exp(-1*E325/R/T) * RADIOM2**1.0 * LIG**1.0  +1*A346 * T**n346 * np.exp(-1*E346/R/T) * RADIOM2**1.0 * PFET3**1.0  +1*A367 * T**n367 * np.exp(-1*E367/R/T) * RADIOM2**1.0 * ADIO**1.0  +1*A388 * T**n388 * np.exp(-1*E388/R/T) * RADIOM2**1.0 * KET**1.0, 
			+1*A12 * T**n12 * np.exp(-1*E12/R/T) * RLIGH**1.0  +1*A13 * T**n13 * np.exp(-1*E13/R/T) * PRLIGH2**1.0, 
			+0.5*A11 * T**n11 * np.exp(-1*E11/R/T) * RPHENOX**1.0  +0.5*A29 * T**n29 * np.exp(-1*E29/R/T) * C10H2M2**1.0  +0.5*A36 * T**n36 * np.exp(-1*E36/R/T) * ADIOM2**1.0 * RPHENOX**1.0  +0.5*A37 * T**n37 * np.exp(-1*E37/R/T) * KETM2**1.0 * RPHENOX**1.0  +0.5*A38 * T**n38 * np.exp(-1*E38/R/T) * KETDM2**1.0 * RPHENOX**1.0  +0.5*A39 * T**n39 * np.exp(-1*E39/R/T) * SYNAPYL**1.0 * RPHENOX**1.0  +1*A40 * T**n40 * np.exp(-1*E40/R/T) * ADIO**1.0 * RPHENOX**1.0  +1*A41 * T**n41 * np.exp(-1*E41/R/T) * KET**1.0 * RPHENOX**1.0  +1*A42 * T**n42 * np.exp(-1*E42/R/T) * KETD**1.0 * RPHENOX**1.0  +1*A43 * T**n43 * np.exp(-1*E43/R/T) * COUMARYL**1.0 * RPHENOX**1.0  +0.5*A44 * T**n44 * np.exp(-1*E44/R/T) * ADIO**1.0 * RPHENOXM2**1.0  +0.5*A45 * T**n45 * np.exp(-1*E45/R/T) * KET**1.0 * RPHENOXM2**1.0  +0.5*A46 * T**n46 * np.exp(-1*E46/R/T) * KETD**1.0 * RPHENOXM2**1.0  +0.5*A47 * T**n47 * np.exp(-1*E47/R/T) * COUMARYL**1.0 * RPHENOXM2**1.0  +0.5*A49 * T**n49 * np.exp(-1*E49/R/T) * C10H2M2**1.0 * RPHENOXM2**1.0  +0.5*A50 * T**n50 * np.exp(-1*E50/R/T) * C10H2M4**1.0 * RPHENOX**1.0  +1*A51 * T**n51 * np.exp(-1*E51/R/T) * C10H2M2**1.0 * RPHENOX**1.0  +0.5*A52 * T**n52 * np.exp(-1*E52/R/T) * RCH3O**1.0 * RPHENOX**1.0  +0.5*A55 * T**n55 * np.exp(-1*E55/R/T) * RPHENOX**1.0 * RCH3**1.0  +1.5*A73 * T**n73 * np.exp(-1*E73/R/T) * RPHENOX**1.0 * RLIGB**1.0  +1*A74 * T**n74 * np.exp(-1*E74/R/T) * RADIO**2.0  +2*A75 * T**n75 * np.exp(-1*E75/R/T) * RLIGB**1.0 * RLIGB**1.0  +2*A76 * T**n76 * np.exp(-1*E76/R/T) * RLIGA**2.0  +1*A77 * T**n77 * np.exp(-1*E77/R/T) * RKET**2.0  +1*A78 * T**n78 * np.exp(-1*E78/R/T) * PRFET3**1.0 * PRFET3**1.0  +1*A80 * T**n80 * np.exp(-1*E80/R/T) * RPHENOX**1.0 * RPHENOL**1.0  +0.5*A81 * T**n81 * np.exp(-1*E81/R/T) * RPHENOX**1.0 * RC3H3O**1.0  +0.5*A82 * T**n82 * np.exp(-1*E82/R/T) * RPHENOX**1.0 * CHAR**1.0  +1*A109 * T**n109 * np.exp(-1*E109/R/T) * PRADIO**2.0  -1*A280 * T**n280 * np.exp(-1*E280/R/T) * RC3H5O2**1.0 * C10H2**1.0  +0.5*A280 * T**n280 * np.exp(-1*E280/R/T) * RC3H5O2**1.0 * C10H2**1.0  -1*A281 * T**n281 * np.exp(-1*E281/R/T) * PRFET3**1.0 * C10H2**1.0  +0.5*A281 * T**n281 * np.exp(-1*E281/R/T) * PRFET3**1.0 * C10H2**1.0  -1*A282 * T**n282 * np.exp(-1*E282/R/T) * RC3H7O2**1.0 * C10H2**1.0  +0.5*A282 * T**n282 * np.exp(-1*E282/R/T) * RC3H7O2**1.0 * C10H2**1.0  -1*A283 * T**n283 * np.exp(-1*E283/R/T) * RADIOM2**1.0 * C10H2**1.0  +0.5*A283 * T**n283 * np.exp(-1*E283/R/T) * RADIOM2**1.0 * C10H2**1.0  -1*A284 * T**n284 * np.exp(-1*E284/R/T) * PRFET3M2**1.0 * C10H2**1.0  +0.5*A284 * T**n284 * np.exp(-1*E284/R/T) * PRFET3M2**1.0 * C10H2**1.0  -1*A285 * T**n285 * np.exp(-1*E285/R/T) * PRLIGH**1.0 * C10H2**1.0  +0.5*A285 * T**n285 * np.exp(-1*E285/R/T) * PRLIGH**1.0 * C10H2**1.0  -1*A286 * T**n286 * np.exp(-1*E286/R/T) * RLIGM2B**1.0 * C10H2**1.0  +0.5*A286 * T**n286 * np.exp(-1*E286/R/T) * RLIGM2B**1.0 * C10H2**1.0  -1*A287 * T**n287 * np.exp(-1*E287/R/T) * RLIGM2A**1.0 * C10H2**1.0  +0.5*A287 * T**n287 * np.exp(-1*E287/R/T) * RLIGM2A**1.0 * C10H2**1.0  -1*A288 * T**n288 * np.exp(-1*E288/R/T) * RCH3**1.0 * C10H2**1.0  +0.5*A288 * T**n288 * np.exp(-1*E288/R/T) * RCH3**1.0 * C10H2**1.0  -1*A289 * T**n289 * np.exp(-1*E289/R/T) * PRKETM2**1.0 * C10H2**1.0  +0.5*A289 * T**n289 * np.exp(-1*E289/R/T) * PRKETM2**1.0 * C10H2**1.0  -1*A290 * T**n290 * np.exp(-1*E290/R/T) * RKET**1.0 * C10H2**1.0  +0.5*A290 * T**n290 * np.exp(-1*E290/R/T) * RKET**1.0 * C10H2**1.0  -1*A291 * T**n291 * np.exp(-1*E291/R/T) * PRADIO**1.0 * C10H2**1.0  +0.5*A291 * T**n291 * np.exp(-1*E291/R/T) * PRADIO**1.0 * C10H2**1.0  -1*A292 * T**n292 * np.exp(-1*E292/R/T) * RC3H3O**1.0 * C10H2**1.0  +0.5*A292 * T**n292 * np.exp(-1*E292/R/T) * RC3H3O**1.0 * C10H2**1.0  -1*A293 * T**n293 * np.exp(-1*E293/R/T) * RLIGB**1.0 * C10H2**1.0  +0.5*A293 * T**n293 * np.exp(-1*E293/R/T) * RLIGB**1.0 * C10H2**1.0  -1*A294 * T**n294 * np.exp(-1*E294/R/T) * RLIGA**1.0 * C10H2**1.0  +0.5*A294 * T**n294 * np.exp(-1*E294/R/T) * RLIGA**1.0 * C10H2**1.0  -1*A295 * T**n295 * np.exp(-1*E295/R/T) * PRADIOM2**1.0 * C10H2**1.0  +0.5*A295 * T**n295 * np.exp(-1*E295/R/T) * PRADIOM2**1.0 * C10H2**1.0  -1*A296 * T**n296 * np.exp(-1*E296/R/T) * RMGUAI**1.0 * C10H2**1.0  +0.5*A296 * T**n296 * np.exp(-1*E296/R/T) * RMGUAI**1.0 * C10H2**1.0  -1*A297 * T**n297 * np.exp(-1*E297/R/T) * OH**1.0 * C10H2**1.0  +0.5*A297 * T**n297 * np.exp(-1*E297/R/T) * OH**1.0 * C10H2**1.0  -1*A298 * T**n298 * np.exp(-1*E298/R/T) * RCH3O**1.0 * C10H2**1.0  +0.5*A298 * T**n298 * np.exp(-1*E298/R/T) * RCH3O**1.0 * C10H2**1.0  -1*A299 * T**n299 * np.exp(-1*E299/R/T) * RPHENOL**1.0 * C10H2**1.0  +0.5*A299 * T**n299 * np.exp(-1*E299/R/T) * RPHENOL**1.0 * C10H2**1.0  -1*A300 * T**n300 * np.exp(-1*E300/R/T) * RADIO**1.0 * C10H2**1.0  +0.5*A300 * T**n300 * np.exp(-1*E300/R/T) * RADIO**1.0 * C10H2**1.0, 
			-1*A29 * T**n29 * np.exp(-1*E29/R/T) * C10H2M2**1.0  +0.5*A48 * T**n48 * np.exp(-1*E48/R/T) * C10H2M4**1.0 * RPHENOXM2**1.0  -1*A49 * T**n49 * np.exp(-1*E49/R/T) * C10H2M2**1.0 * RPHENOXM2**1.0  +0.5*A49 * T**n49 * np.exp(-1*E49/R/T) * C10H2M2**1.0 * RPHENOXM2**1.0  +0.5*A50 * T**n50 * np.exp(-1*E50/R/T) * C10H2M4**1.0 * RPHENOX**1.0  -1*A51 * T**n51 * np.exp(-1*E51/R/T) * C10H2M2**1.0 * RPHENOX**1.0  +0.5*A51 * T**n51 * np.exp(-1*E51/R/T) * C10H2M2**1.0 * RPHENOX**1.0, 
			+0.5*A10 * T**n10 * np.exp(-1*E10/R/T) * RPHENOXM2**1.0  -1*A28 * T**n28 * np.exp(-1*E28/R/T) * C10H2M4**1.0  +0.5*A28 * T**n28 * np.exp(-1*E28/R/T) * C10H2M4**1.0  +1*A32 * T**n32 * np.exp(-1*E32/R/T) * ADIOM2**1.0 * RPHENOXM2**1.0  +1*A33 * T**n33 * np.exp(-1*E33/R/T) * KETM2**1.0 * RPHENOXM2**1.0  +1*A34 * T**n34 * np.exp(-1*E34/R/T) * KETDM2**1.0 * RPHENOXM2**1.0  +1*A35 * T**n35 * np.exp(-1*E35/R/T) * SYNAPYL**1.0 * RPHENOXM2**1.0  +0.5*A36 * T**n36 * np.exp(-1*E36/R/T) * ADIOM2**1.0 * RPHENOX**1.0  +0.5*A37 * T**n37 * np.exp(-1*E37/R/T) * KETM2**1.0 * RPHENOX**1.0  +0.5*A38 * T**n38 * np.exp(-1*E38/R/T) * KETDM2**1.0 * RPHENOX**1.0  +0.5*A39 * T**n39 * np.exp(-1*E39/R/T) * SYNAPYL**1.0 * RPHENOX**1.0  +0.5*A44 * T**n44 * np.exp(-1*E44/R/T) * ADIO**1.0 * RPHENOXM2**1.0  +0.5*A45 * T**n45 * np.exp(-1*E45/R/T) * KET**1.0 * RPHENOXM2**1.0  +0.5*A46 * T**n46 * np.exp(-1*E46/R/T) * KETD**1.0 * RPHENOXM2**1.0  +0.5*A47 * T**n47 * np.exp(-1*E47/R/T) * COUMARYL**1.0 * RPHENOXM2**1.0  -1*A48 * T**n48 * np.exp(-1*E48/R/T) * C10H2M4**1.0 * RPHENOXM2**1.0  +1*A48 * T**n48 * np.exp(-1*E48/R/T) * C10H2M4**1.0 * RPHENOXM2**1.0  +0.5*A49 * T**n49 * np.exp(-1*E49/R/T) * C10H2M2**1.0 * RPHENOXM2**1.0  -1*A50 * T**n50 * np.exp(-1*E50/R/T) * C10H2M4**1.0 * RPHENOX**1.0  +0.5*A50 * T**n50 * np.exp(-1*E50/R/T) * C10H2M4**1.0 * RPHENOX**1.0  +0.5*A53 * T**n53 * np.exp(-1*E53/R/T) * RCH3O**1.0 * RPHENOXM2**1.0  +0.5*A54 * T**n54 * np.exp(-1*E54/R/T) * RPHENOXM2**1.0 * RCH3**1.0  +1*A60 * T**n60 * np.exp(-1*E60/R/T) * RADIOM2**2.0  +2*A61 * T**n61 * np.exp(-1*E61/R/T) * RLIGM2B**2.0  +2*A62 * T**n62 * np.exp(-1*E62/R/T) * RLIGM2A**2.0  +1*A63 * T**n63 * np.exp(-1*E63/R/T) * RMGUAI**2.0  +1*A64 * T**n64 * np.exp(-1*E64/R/T) * RKETM2**2.0  +1*A65 * T**n65 * np.exp(-1*E65/R/T) * PRFET3M2**1.0 * PRFET3M2**1.0  +2*A79 * T**n79 * np.exp(-1*E79/R/T) * RLIGH**1.0 * RLIGH**1.0  +0.5*A83 * T**n83 * np.exp(-1*E83/R/T) * RPHENOXM2**1.0 * CHAR**1.0  +1*A110 * T**n110 * np.exp(-1*E110/R/T) * PRADIOM2**2.0, 
			+1*A70 * T**n70 * np.exp(-1*E70/R/T) * RCH3**2.0, 
			+1*A124 * T**n124 * np.exp(-1*E124/R/T) * RC3H3O**1.0 * LIGH**1.0  +1*A145 * T**n145 * np.exp(-1*E145/R/T) * RC3H3O**1.0 * PLIGH**1.0  +1*A166 * T**n166 * np.exp(-1*E166/R/T) * RC3H3O**1.0 * PLIGM2**1.0  +1*A187 * T**n187 * np.exp(-1*E187/R/T) * RC3H3O**1.0 * LIGM2**1.0  +1*A208 * T**n208 * np.exp(-1*E208/R/T) * RC3H3O**1.0 * LIGM2**1.0  +1*A229 * T**n229 * np.exp(-1*E229/R/T) * RC3H3O**1.0 * PFET3M2**1.0  +1*A250 * T**n250 * np.exp(-1*E250/R/T) * RC3H3O**1.0 * ADIOM2**1.0  +1*A271 * T**n271 * np.exp(-1*E271/R/T) * RC3H3O**1.0 * KETM2**1.0  +1*A292 * T**n292 * np.exp(-1*E292/R/T) * RC3H3O**1.0 * C10H2**1.0  +1*A313 * T**n313 * np.exp(-1*E313/R/T) * RC3H3O**1.0 * LIG**1.0  +1*A334 * T**n334 * np.exp(-1*E334/R/T) * RC3H3O**1.0 * LIG**1.0  +1*A355 * T**n355 * np.exp(-1*E355/R/T) * RC3H3O**1.0 * PFET3**1.0  +1*A376 * T**n376 * np.exp(-1*E376/R/T) * RC3H3O**1.0 * ADIO**1.0  +1*A397 * T**n397 * np.exp(-1*E397/R/T) * RC3H3O**1.0 * KET**1.0, 
			+1*A18 * T**n18 * np.exp(-1*E18/R/T) * PRFET3M2**1.0  +1*A22 * T**n22 * np.exp(-1*E22/R/T) * PRFET3**1.0, 
			+1*A1 * T**n1 * np.exp(-1*E1/R/T) * LIGH**1.0, 
			+1*A14 * T**n14 * np.exp(-1*E14/R/T) * RADIOM2**1.0  +1*A19 * T**n19 * np.exp(-1*E19/R/T) * RADIO**1.0  +1*A112 * T**n112 * np.exp(-1*E112/R/T) * RC3H5O2**1.0 * LIGH**1.0  +1*A133 * T**n133 * np.exp(-1*E133/R/T) * RC3H5O2**1.0 * PLIGH**1.0  +1*A154 * T**n154 * np.exp(-1*E154/R/T) * RC3H5O2**1.0 * PLIGM2**1.0  +1*A175 * T**n175 * np.exp(-1*E175/R/T) * RC3H5O2**1.0 * LIGM2**1.0  +1*A196 * T**n196 * np.exp(-1*E196/R/T) * RC3H5O2**1.0 * LIGM2**1.0  +1*A217 * T**n217 * np.exp(-1*E217/R/T) * RC3H5O2**1.0 * PFET3M2**1.0  +1*A238 * T**n238 * np.exp(-1*E238/R/T) * RC3H5O2**1.0 * ADIOM2**1.0  +1*A259 * T**n259 * np.exp(-1*E259/R/T) * RC3H5O2**1.0 * KETM2**1.0  +1*A280 * T**n280 * np.exp(-1*E280/R/T) * RC3H5O2**1.0 * C10H2**1.0  +1*A301 * T**n301 * np.exp(-1*E301/R/T) * RC3H5O2**1.0 * LIG**1.0  +1*A322 * T**n322 * np.exp(-1*E322/R/T) * RC3H5O2**1.0 * LIG**1.0  +1*A343 * T**n343 * np.exp(-1*E343/R/T) * RC3H5O2**1.0 * PFET3**1.0  +1*A364 * T**n364 * np.exp(-1*E364/R/T) * RC3H5O2**1.0 * ADIO**1.0  +1*A385 * T**n385 * np.exp(-1*E385/R/T) * RC3H5O2**1.0 * KET**1.0, 
			+1*A114 * T**n114 * np.exp(-1*E114/R/T) * RC3H7O2**1.0 * LIGH**1.0  +1*A135 * T**n135 * np.exp(-1*E135/R/T) * RC3H7O2**1.0 * PLIGH**1.0  +1*A156 * T**n156 * np.exp(-1*E156/R/T) * RC3H7O2**1.0 * PLIGM2**1.0  +1*A177 * T**n177 * np.exp(-1*E177/R/T) * RC3H7O2**1.0 * LIGM2**1.0  +1*A198 * T**n198 * np.exp(-1*E198/R/T) * RC3H7O2**1.0 * LIGM2**1.0  +1*A219 * T**n219 * np.exp(-1*E219/R/T) * RC3H7O2**1.0 * PFET3M2**1.0  +1*A240 * T**n240 * np.exp(-1*E240/R/T) * RC3H7O2**1.0 * ADIOM2**1.0  +1*A261 * T**n261 * np.exp(-1*E261/R/T) * RC3H7O2**1.0 * KETM2**1.0  +1*A282 * T**n282 * np.exp(-1*E282/R/T) * RC3H7O2**1.0 * C10H2**1.0  +1*A303 * T**n303 * np.exp(-1*E303/R/T) * RC3H7O2**1.0 * LIG**1.0  +1*A324 * T**n324 * np.exp(-1*E324/R/T) * RC3H7O2**1.0 * LIG**1.0  +1*A345 * T**n345 * np.exp(-1*E345/R/T) * RC3H7O2**1.0 * PFET3**1.0  +1*A366 * T**n366 * np.exp(-1*E366/R/T) * RC3H7O2**1.0 * ADIO**1.0  +1*A387 * T**n387 * np.exp(-1*E387/R/T) * RC3H7O2**1.0 * KET**1.0, 
			+1*A30 * T**n30 * np.exp(-1*E30/R/T) * PLIGC**1.0  +1*A108 * T**n108 * np.exp(-1*E108/R/T) * LIGC**1.0, 
			+1*A27 * T**n27 * np.exp(-1*E27/R/T) * RC3H7O2**1.0  +1*A57 * T**n57 * np.exp(-1*E57/R/T) * RCH3O**2.0, 
			+1*A69 * T**n69 * np.exp(-1*E69/R/T) * OH**1.0 * RCH3**1.0  +1*A130 * T**n130 * np.exp(-1*E130/R/T) * RCH3O**1.0 * LIGH**1.0  +1*A151 * T**n151 * np.exp(-1*E151/R/T) * RCH3O**1.0 * PLIGH**1.0  +1*A172 * T**n172 * np.exp(-1*E172/R/T) * RCH3O**1.0 * PLIGM2**1.0  +1*A193 * T**n193 * np.exp(-1*E193/R/T) * RCH3O**1.0 * LIGM2**1.0  +1*A214 * T**n214 * np.exp(-1*E214/R/T) * RCH3O**1.0 * LIGM2**1.0  +1*A235 * T**n235 * np.exp(-1*E235/R/T) * RCH3O**1.0 * PFET3M2**1.0  +1*A256 * T**n256 * np.exp(-1*E256/R/T) * RCH3O**1.0 * ADIOM2**1.0  +1*A277 * T**n277 * np.exp(-1*E277/R/T) * RCH3O**1.0 * KETM2**1.0  +1*A298 * T**n298 * np.exp(-1*E298/R/T) * RCH3O**1.0 * C10H2**1.0  +1*A319 * T**n319 * np.exp(-1*E319/R/T) * RCH3O**1.0 * LIG**1.0  +1*A340 * T**n340 * np.exp(-1*E340/R/T) * RCH3O**1.0 * LIG**1.0  +1*A361 * T**n361 * np.exp(-1*E361/R/T) * RCH3O**1.0 * PFET3**1.0  +1*A382 * T**n382 * np.exp(-1*E382/R/T) * RCH3O**1.0 * ADIO**1.0  +1*A403 * T**n403 * np.exp(-1*E403/R/T) * RCH3O**1.0 * KET**1.0, 
			+1*A120 * T**n120 * np.exp(-1*E120/R/T) * RCH3**1.0 * LIGH**1.0  +1*A141 * T**n141 * np.exp(-1*E141/R/T) * RCH3**1.0 * PLIGH**1.0  +1*A162 * T**n162 * np.exp(-1*E162/R/T) * RCH3**1.0 * PLIGM2**1.0  +1*A183 * T**n183 * np.exp(-1*E183/R/T) * RCH3**1.0 * LIGM2**1.0  +1*A204 * T**n204 * np.exp(-1*E204/R/T) * RCH3**1.0 * LIGM2**1.0  +1*A225 * T**n225 * np.exp(-1*E225/R/T) * RCH3**1.0 * PFET3M2**1.0  +1*A246 * T**n246 * np.exp(-1*E246/R/T) * RCH3**1.0 * ADIOM2**1.0  +1*A267 * T**n267 * np.exp(-1*E267/R/T) * RCH3**1.0 * KETM2**1.0  +1*A288 * T**n288 * np.exp(-1*E288/R/T) * RCH3**1.0 * C10H2**1.0  +1*A309 * T**n309 * np.exp(-1*E309/R/T) * RCH3**1.0 * LIG**1.0  +1*A330 * T**n330 * np.exp(-1*E330/R/T) * RCH3**1.0 * LIG**1.0  +1*A351 * T**n351 * np.exp(-1*E351/R/T) * RCH3**1.0 * PFET3**1.0  +1*A372 * T**n372 * np.exp(-1*E372/R/T) * RCH3**1.0 * ADIO**1.0  +1*A393 * T**n393 * np.exp(-1*E393/R/T) * RCH3**1.0 * KET**1.0, 
			+0.2*A61 * T**n61 * np.exp(-1*E61/R/T) * RLIGM2B**2.0  +0.2*A75 * T**n75 * np.exp(-1*E75/R/T) * RLIGB**1.0 * RLIGB**1.0  -1*A82 * T**n82 * np.exp(-1*E82/R/T) * RPHENOX**1.0 * CHAR**1.0  +1*A82 * T**n82 * np.exp(-1*E82/R/T) * RPHENOX**1.0 * CHAR**1.0  -1*A83 * T**n83 * np.exp(-1*E83/R/T) * RPHENOXM2**1.0 * CHAR**1.0  +1*A83 * T**n83 * np.exp(-1*E83/R/T) * RPHENOXM2**1.0 * CHAR**1.0  +0.2*A97 * T**n97 * np.exp(-1*E97/R/T) * PC2H2**1.0  +0.1*A100 * T**n100 * np.exp(-1*E100/R/T) * PCOHP2**1.0  +0.1*A103 * T**n103 * np.exp(-1*E103/R/T) * PCHP2**1.0  +0.5*A280 * T**n280 * np.exp(-1*E280/R/T) * RC3H5O2**1.0 * C10H2**1.0  +0.5*A281 * T**n281 * np.exp(-1*E281/R/T) * PRFET3**1.0 * C10H2**1.0  +0.5*A282 * T**n282 * np.exp(-1*E282/R/T) * RC3H7O2**1.0 * C10H2**1.0  +0.5*A283 * T**n283 * np.exp(-1*E283/R/T) * RADIOM2**1.0 * C10H2**1.0  +0.5*A284 * T**n284 * np.exp(-1*E284/R/T) * PRFET3M2**1.0 * C10H2**1.0  +0.5*A285 * T**n285 * np.exp(-1*E285/R/T) * PRLIGH**1.0 * C10H2**1.0  +0.5*A286 * T**n286 * np.exp(-1*E286/R/T) * RLIGM2B**1.0 * C10H2**1.0  +0.5*A287 * T**n287 * np.exp(-1*E287/R/T) * RLIGM2A**1.0 * C10H2**1.0  +0.5*A288 * T**n288 * np.exp(-1*E288/R/T) * RCH3**1.0 * C10H2**1.0  +0.5*A289 * T**n289 * np.exp(-1*E289/R/T) * PRKETM2**1.0 * C10H2**1.0  +0.5*A290 * T**n290 * np.exp(-1*E290/R/T) * RKET**1.0 * C10H2**1.0  +0.5*A291 * T**n291 * np.exp(-1*E291/R/T) * PRADIO**1.0 * C10H2**1.0  +0.5*A292 * T**n292 * np.exp(-1*E292/R/T) * RC3H3O**1.0 * C10H2**1.0  +0.5*A293 * T**n293 * np.exp(-1*E293/R/T) * RLIGB**1.0 * C10H2**1.0  +0.5*A294 * T**n294 * np.exp(-1*E294/R/T) * RLIGA**1.0 * C10H2**1.0  +0.5*A295 * T**n295 * np.exp(-1*E295/R/T) * PRADIOM2**1.0 * C10H2**1.0  +0.5*A296 * T**n296 * np.exp(-1*E296/R/T) * RMGUAI**1.0 * C10H2**1.0  +0.5*A297 * T**n297 * np.exp(-1*E297/R/T) * OH**1.0 * C10H2**1.0  +0.5*A298 * T**n298 * np.exp(-1*E298/R/T) * RCH3O**1.0 * C10H2**1.0  +0.5*A299 * T**n299 * np.exp(-1*E299/R/T) * RPHENOL**1.0 * C10H2**1.0  +0.5*A300 * T**n300 * np.exp(-1*E300/R/T) * RADIO**1.0 * C10H2**1.0, 
			+1*A10 * T**n10 * np.exp(-1*E10/R/T) * RPHENOXM2**1.0  +1*A11 * T**n11 * np.exp(-1*E11/R/T) * RPHENOX**1.0  +1*A94 * T**n94 * np.exp(-1*E94/R/T) * PCOS**1.0  +1*A95 * T**n95 * np.exp(-1*E95/R/T) * PCOH**1.0  +1*A111 * T**n111 * np.exp(-1*E111/R/T) * PCHO**1.0, 
			+1*A31 * T**n31 * np.exp(-1*E31/R/T) * PLIGO**1.0  +1*A52 * T**n52 * np.exp(-1*E52/R/T) * RCH3O**1.0 * RPHENOX**1.0  +1*A53 * T**n53 * np.exp(-1*E53/R/T) * RCH3O**1.0 * RPHENOXM2**1.0  +1*A107 * T**n107 * np.exp(-1*E107/R/T) * LIGO**1.0, 
			+1*A25 * T**n25 * np.exp(-1*E25/R/T) * RADIO**1.0  -1*A43 * T**n43 * np.exp(-1*E43/R/T) * COUMARYL**1.0 * RPHENOX**1.0  -1*A47 * T**n47 * np.exp(-1*E47/R/T) * COUMARYL**1.0 * RPHENOXM2**1.0  -1*A89 * T**n89 * np.exp(-1*E89/R/T) * COUMARYL**1.0, 
			+1*A56 * T**n56 * np.exp(-1*E56/R/T) * RCH3O**1.0 * RCH3**1.0, 
			+0.5*A10 * T**n10 * np.exp(-1*E10/R/T) * RPHENOXM2**1.0  +1.5*A11 * T**n11 * np.exp(-1*E11/R/T) * RPHENOX**1.0  +0.5*A32 * T**n32 * np.exp(-1*E32/R/T) * ADIOM2**1.0 * RPHENOXM2**1.0  +0.5*A33 * T**n33 * np.exp(-1*E33/R/T) * KETM2**1.0 * RPHENOXM2**1.0  +0.5*A34 * T**n34 * np.exp(-1*E34/R/T) * KETDM2**1.0 * RPHENOXM2**1.0  +1*A35 * T**n35 * np.exp(-1*E35/R/T) * SYNAPYL**1.0 * RPHENOXM2**1.0  +0.5*A37 * T**n37 * np.exp(-1*E37/R/T) * KETM2**1.0 * RPHENOX**1.0  +0.5*A38 * T**n38 * np.exp(-1*E38/R/T) * KETDM2**1.0 * RPHENOX**1.0  +1.5*A39 * T**n39 * np.exp(-1*E39/R/T) * SYNAPYL**1.0 * RPHENOX**1.0  +1.5*A40 * T**n40 * np.exp(-1*E40/R/T) * ADIO**1.0 * RPHENOX**1.0  +1.5*A41 * T**n41 * np.exp(-1*E41/R/T) * KET**1.0 * RPHENOX**1.0  +1.5*A42 * T**n42 * np.exp(-1*E42/R/T) * KETD**1.0 * RPHENOX**1.0  +2.5*A43 * T**n43 * np.exp(-1*E43/R/T) * COUMARYL**1.0 * RPHENOX**1.0  +1.5*A44 * T**n44 * np.exp(-1*E44/R/T) * ADIO**1.0 * RPHENOXM2**1.0  +1.5*A45 * T**n45 * np.exp(-1*E45/R/T) * KET**1.0 * RPHENOXM2**1.0  +1.5*A46 * T**n46 * np.exp(-1*E46/R/T) * KETD**1.0 * RPHENOXM2**1.0  +1.5*A47 * T**n47 * np.exp(-1*E47/R/T) * COUMARYL**1.0 * RPHENOXM2**1.0  +0.5*A48 * T**n48 * np.exp(-1*E48/R/T) * C10H2M4**1.0 * RPHENOXM2**1.0  +0.5*A49 * T**n49 * np.exp(-1*E49/R/T) * C10H2M2**1.0 * RPHENOXM2**1.0  +1.5*A50 * T**n50 * np.exp(-1*E50/R/T) * C10H2M4**1.0 * RPHENOX**1.0  +1.5*A51 * T**n51 * np.exp(-1*E51/R/T) * C10H2M2**1.0 * RPHENOX**1.0  +1.5*A52 * T**n52 * np.exp(-1*E52/R/T) * RCH3O**1.0 * RPHENOX**1.0  +0.5*A53 * T**n53 * np.exp(-1*E53/R/T) * RCH3O**1.0 * RPHENOXM2**1.0  +2*A73 * T**n73 * np.exp(-1*E73/R/T) * RPHENOX**1.0 * RLIGB**1.0  +1*A74 * T**n74 * np.exp(-1*E74/R/T) * RADIO**2.0  +2*A75 * T**n75 * np.exp(-1*E75/R/T) * RLIGB**1.0 * RLIGB**1.0  +2*A76 * T**n76 * np.exp(-1*E76/R/T) * RLIGA**2.0  +2*A77 * T**n77 * np.exp(-1*E77/R/T) * RKET**2.0  +1*A78 * T**n78 * np.exp(-1*E78/R/T) * PRFET3**1.0 * PRFET3**1.0  +1*A79 * T**n79 * np.exp(-1*E79/R/T) * RLIGH**1.0 * RLIGH**1.0  +1.5*A80 * T**n80 * np.exp(-1*E80/R/T) * RPHENOX**1.0 * RPHENOL**1.0  +1*A81 * T**n81 * np.exp(-1*E81/R/T) * RPHENOX**1.0 * RC3H3O**1.0  +1.5*A82 * T**n82 * np.exp(-1*E82/R/T) * RPHENOX**1.0 * CHAR**1.0  +0.5*A83 * T**n83 * np.exp(-1*E83/R/T) * RPHENOXM2**1.0 * CHAR**1.0  +1*A96 * T**n96 * np.exp(-1*E96/R/T) * PH2**1.0  +1*A97 * T**n97 * np.exp(-1*E97/R/T) * PC2H2**1.0  +0.5*A102 * T**n102 * np.exp(-1*E102/R/T) * PCH2P**1.0  +0.5*A103 * T**n103 * np.exp(-1*E103/R/T) * PCHP2**1.0  +1*A109 * T**n109 * np.exp(-1*E109/R/T) * PRADIO**2.0, 
			+1*A57 * T**n57 * np.exp(-1*E57/R/T) * RCH3O**2.0  +2*A62 * T**n62 * np.exp(-1*E62/R/T) * RLIGM2A**2.0  +2*A66 * T**n66 * np.exp(-1*E66/R/T) * RC3H7O2**2.0  +2*A67 * T**n67 * np.exp(-1*E67/R/T) * RC3H5O2**2.0  +1*A73 * T**n73 * np.exp(-1*E73/R/T) * RPHENOX**1.0 * RLIGB**1.0  +2*A76 * T**n76 * np.exp(-1*E76/R/T) * RLIGA**2.0  +4*A79 * T**n79 * np.exp(-1*E79/R/T) * RLIGH**1.0 * RLIGH**1.0  +1*A129 * T**n129 * np.exp(-1*E129/R/T) * OH**1.0 * LIGH**1.0  +1*A150 * T**n150 * np.exp(-1*E150/R/T) * OH**1.0 * PLIGH**1.0  +1*A171 * T**n171 * np.exp(-1*E171/R/T) * OH**1.0 * PLIGM2**1.0  +1*A192 * T**n192 * np.exp(-1*E192/R/T) * OH**1.0 * LIGM2**1.0  +1*A213 * T**n213 * np.exp(-1*E213/R/T) * OH**1.0 * LIGM2**1.0  +1*A234 * T**n234 * np.exp(-1*E234/R/T) * OH**1.0 * PFET3M2**1.0  +1*A255 * T**n255 * np.exp(-1*E255/R/T) * OH**1.0 * ADIOM2**1.0  +1*A276 * T**n276 * np.exp(-1*E276/R/T) * OH**1.0 * KETM2**1.0  +1*A297 * T**n297 * np.exp(-1*E297/R/T) * OH**1.0 * C10H2**1.0  +1*A318 * T**n318 * np.exp(-1*E318/R/T) * OH**1.0 * LIG**1.0  +1*A339 * T**n339 * np.exp(-1*E339/R/T) * OH**1.0 * LIG**1.0  +1*A360 * T**n360 * np.exp(-1*E360/R/T) * OH**1.0 * PFET3**1.0  +1*A381 * T**n381 * np.exp(-1*E381/R/T) * OH**1.0 * ADIO**1.0  +1*A402 * T**n402 * np.exp(-1*E402/R/T) * OH**1.0 * KET**1.0, 
			+1*A20 * T**n20 * np.exp(-1*E20/R/T) * RLIGA**1.0  -1*A41 * T**n41 * np.exp(-1*E41/R/T) * KET**1.0 * RPHENOX**1.0  -1*A45 * T**n45 * np.exp(-1*E45/R/T) * KET**1.0 * RPHENOXM2**1.0  -1*A91 * T**n91 * np.exp(-1*E91/R/T) * KET**1.0  +1*A122 * T**n122 * np.exp(-1*E122/R/T) * RKET**1.0 * LIGH**1.0  +1*A143 * T**n143 * np.exp(-1*E143/R/T) * RKET**1.0 * PLIGH**1.0  +1*A164 * T**n164 * np.exp(-1*E164/R/T) * RKET**1.0 * PLIGM2**1.0  +1*A185 * T**n185 * np.exp(-1*E185/R/T) * RKET**1.0 * LIGM2**1.0  +1*A206 * T**n206 * np.exp(-1*E206/R/T) * RKET**1.0 * LIGM2**1.0  +1*A227 * T**n227 * np.exp(-1*E227/R/T) * RKET**1.0 * PFET3M2**1.0  +1*A248 * T**n248 * np.exp(-1*E248/R/T) * RKET**1.0 * ADIOM2**1.0  +1*A269 * T**n269 * np.exp(-1*E269/R/T) * RKET**1.0 * KETM2**1.0  +1*A290 * T**n290 * np.exp(-1*E290/R/T) * RKET**1.0 * C10H2**1.0  +1*A311 * T**n311 * np.exp(-1*E311/R/T) * RKET**1.0 * LIG**1.0  +1*A332 * T**n332 * np.exp(-1*E332/R/T) * RKET**1.0 * LIG**1.0  +1*A353 * T**n353 * np.exp(-1*E353/R/T) * RKET**1.0 * PFET3**1.0  +1*A374 * T**n374 * np.exp(-1*E374/R/T) * RKET**1.0 * ADIO**1.0  -1*A385 * T**n385 * np.exp(-1*E385/R/T) * RC3H5O2**1.0 * KET**1.0  -1*A386 * T**n386 * np.exp(-1*E386/R/T) * PRFET3**1.0 * KET**1.0  -1*A387 * T**n387 * np.exp(-1*E387/R/T) * RC3H7O2**1.0 * KET**1.0  -1*A388 * T**n388 * np.exp(-1*E388/R/T) * RADIOM2**1.0 * KET**1.0  -1*A389 * T**n389 * np.exp(-1*E389/R/T) * PRFET3M2**1.0 * KET**1.0  -1*A390 * T**n390 * np.exp(-1*E390/R/T) * PRLIGH**1.0 * KET**1.0  -1*A391 * T**n391 * np.exp(-1*E391/R/T) * RLIGM2B**1.0 * KET**1.0  -1*A392 * T**n392 * np.exp(-1*E392/R/T) * RLIGM2A**1.0 * KET**1.0  -1*A393 * T**n393 * np.exp(-1*E393/R/T) * RCH3**1.0 * KET**1.0  -1*A394 * T**n394 * np.exp(-1*E394/R/T) * PRKETM2**1.0 * KET**1.0  -1*A395 * T**n395 * np.exp(-1*E395/R/T) * RKET**1.0 * KET**1.0  +1*A395 * T**n395 * np.exp(-1*E395/R/T) * RKET**1.0 * KET**1.0  -1*A396 * T**n396 * np.exp(-1*E396/R/T) * PRADIO**1.0 * KET**1.0  -1*A397 * T**n397 * np.exp(-1*E397/R/T) * RC3H3O**1.0 * KET**1.0  -1*A398 * T**n398 * np.exp(-1*E398/R/T) * RLIGB**1.0 * KET**1.0  -1*A399 * T**n399 * np.exp(-1*E399/R/T) * RLIGA**1.0 * KET**1.0  -1*A400 * T**n400 * np.exp(-1*E400/R/T) * PRADIOM2**1.0 * KET**1.0  -1*A401 * T**n401 * np.exp(-1*E401/R/T) * RMGUAI**1.0 * KET**1.0  -1*A402 * T**n402 * np.exp(-1*E402/R/T) * OH**1.0 * KET**1.0  -1*A403 * T**n403 * np.exp(-1*E403/R/T) * RCH3O**1.0 * KET**1.0  -1*A404 * T**n404 * np.exp(-1*E404/R/T) * RPHENOL**1.0 * KET**1.0  -1*A405 * T**n405 * np.exp(-1*E405/R/T) * RADIO**1.0 * KET**1.0, 
			+1*A26 * T**n26 * np.exp(-1*E26/R/T) * RKET**1.0  -1*A42 * T**n42 * np.exp(-1*E42/R/T) * KETD**1.0 * RPHENOX**1.0  -1*A46 * T**n46 * np.exp(-1*E46/R/T) * KETD**1.0 * RPHENOXM2**1.0  -1*A92 * T**n92 * np.exp(-1*E92/R/T) * KETD**1.0, 
			+1*A9 * T**n9 * np.exp(-1*E9/R/T) * PRKETM2**1.0  +1*A24 * T**n24 * np.exp(-1*E24/R/T) * RKETM2**1.0  -1*A34 * T**n34 * np.exp(-1*E34/R/T) * KETDM2**1.0 * RPHENOXM2**1.0  -1*A38 * T**n38 * np.exp(-1*E38/R/T) * KETDM2**1.0 * RPHENOX**1.0  -1*A86 * T**n86 * np.exp(-1*E86/R/T) * KETDM2**1.0, 
			+1*A15 * T**n15 * np.exp(-1*E15/R/T) * RLIGM2A**1.0  -1*A33 * T**n33 * np.exp(-1*E33/R/T) * KETM2**1.0 * RPHENOXM2**1.0  -1*A37 * T**n37 * np.exp(-1*E37/R/T) * KETM2**1.0 * RPHENOX**1.0  -1*A85 * T**n85 * np.exp(-1*E85/R/T) * KETM2**1.0  -1*A259 * T**n259 * np.exp(-1*E259/R/T) * RC3H5O2**1.0 * KETM2**1.0  -1*A260 * T**n260 * np.exp(-1*E260/R/T) * PRFET3**1.0 * KETM2**1.0  -1*A261 * T**n261 * np.exp(-1*E261/R/T) * RC3H7O2**1.0 * KETM2**1.0  -1*A262 * T**n262 * np.exp(-1*E262/R/T) * RADIOM2**1.0 * KETM2**1.0  -1*A263 * T**n263 * np.exp(-1*E263/R/T) * PRFET3M2**1.0 * KETM2**1.0  -1*A264 * T**n264 * np.exp(-1*E264/R/T) * PRLIGH**1.0 * KETM2**1.0  -1*A265 * T**n265 * np.exp(-1*E265/R/T) * RLIGM2B**1.0 * KETM2**1.0  -1*A266 * T**n266 * np.exp(-1*E266/R/T) * RLIGM2A**1.0 * KETM2**1.0  -1*A267 * T**n267 * np.exp(-1*E267/R/T) * RCH3**1.0 * KETM2**1.0  -1*A268 * T**n268 * np.exp(-1*E268/R/T) * PRKETM2**1.0 * KETM2**1.0  -1*A269 * T**n269 * np.exp(-1*E269/R/T) * RKET**1.0 * KETM2**1.0  -1*A270 * T**n270 * np.exp(-1*E270/R/T) * PRADIO**1.0 * KETM2**1.0  -1*A271 * T**n271 * np.exp(-1*E271/R/T) * RC3H3O**1.0 * KETM2**1.0  -1*A272 * T**n272 * np.exp(-1*E272/R/T) * RLIGB**1.0 * KETM2**1.0  -1*A273 * T**n273 * np.exp(-1*E273/R/T) * RLIGA**1.0 * KETM2**1.0  -1*A274 * T**n274 * np.exp(-1*E274/R/T) * PRADIOM2**1.0 * KETM2**1.0  -1*A275 * T**n275 * np.exp(-1*E275/R/T) * RMGUAI**1.0 * KETM2**1.0  -1*A276 * T**n276 * np.exp(-1*E276/R/T) * OH**1.0 * KETM2**1.0  -1*A277 * T**n277 * np.exp(-1*E277/R/T) * RCH3O**1.0 * KETM2**1.0  -1*A278 * T**n278 * np.exp(-1*E278/R/T) * RPHENOL**1.0 * KETM2**1.0  -1*A279 * T**n279 * np.exp(-1*E279/R/T) * RADIO**1.0 * KETM2**1.0, 
			-1*A4 * T**n4 * np.exp(-1*E4/R/T) * LIG**1.0  +1*A108 * T**n108 * np.exp(-1*E108/R/T) * LIGC**1.0  +1*A125 * T**n125 * np.exp(-1*E125/R/T) * RLIGB**1.0 * LIGH**1.0  +1*A126 * T**n126 * np.exp(-1*E126/R/T) * RLIGA**1.0 * LIGH**1.0  +1*A146 * T**n146 * np.exp(-1*E146/R/T) * RLIGB**1.0 * PLIGH**1.0  +1*A147 * T**n147 * np.exp(-1*E147/R/T) * RLIGA**1.0 * PLIGH**1.0  +1*A167 * T**n167 * np.exp(-1*E167/R/T) * RLIGB**1.0 * PLIGM2**1.0  +1*A168 * T**n168 * np.exp(-1*E168/R/T) * RLIGA**1.0 * PLIGM2**1.0  +1*A188 * T**n188 * np.exp(-1*E188/R/T) * RLIGB**1.0 * LIGM2**1.0  +1*A189 * T**n189 * np.exp(-1*E189/R/T) * RLIGA**1.0 * LIGM2**1.0  +1*A209 * T**n209 * np.exp(-1*E209/R/T) * RLIGB**1.0 * LIGM2**1.0  +1*A210 * T**n210 * np.exp(-1*E210/R/T) * RLIGA**1.0 * LIGM2**1.0  +1*A230 * T**n230 * np.exp(-1*E230/R/T) * RLIGB**1.0 * PFET3M2**1.0  +1*A231 * T**n231 * np.exp(-1*E231/R/T) * RLIGA**1.0 * PFET3M2**1.0  +1*A251 * T**n251 * np.exp(-1*E251/R/T) * RLIGB**1.0 * ADIOM2**1.0  +1*A252 * T**n252 * np.exp(-1*E252/R/T) * RLIGA**1.0 * ADIOM2**1.0  +1*A272 * T**n272 * np.exp(-1*E272/R/T) * RLIGB**1.0 * KETM2**1.0  +1*A273 * T**n273 * np.exp(-1*E273/R/T) * RLIGA**1.0 * KETM2**1.0  +1*A293 * T**n293 * np.exp(-1*E293/R/T) * RLIGB**1.0 * C10H2**1.0  +1*A294 * T**n294 * np.exp(-1*E294/R/T) * RLIGA**1.0 * C10H2**1.0  -1*A301 * T**n301 * np.exp(-1*E301/R/T) * RC3H5O2**1.0 * LIG**1.0  -1*A302 * T**n302 * np.exp(-1*E302/R/T) * PRFET3**1.0 * LIG**1.0  -1*A303 * T**n303 * np.exp(-1*E303/R/T) * RC3H7O2**1.0 * LIG**1.0  -1*A304 * T**n304 * np.exp(-1*E304/R/T) * RADIOM2**1.0 * LIG**1.0  -1*A305 * T**n305 * np.exp(-1*E305/R/T) * PRFET3M2**1.0 * LIG**1.0  -1*A306 * T**n306 * np.exp(-1*E306/R/T) * PRLIGH**1.0 * LIG**1.0  -1*A307 * T**n307 * np.exp(-1*E307/R/T) * RLIGM2B**1.0 * LIG**1.0  -1*A308 * T**n308 * np.exp(-1*E308/R/T) * RLIGM2A**1.0 * LIG**1.0  -1*A309 * T**n309 * np.exp(-1*E309/R/T) * RCH3**1.0 * LIG**1.0  -1*A310 * T**n310 * np.exp(-1*E310/R/T) * PRKETM2**1.0 * LIG**1.0  -1*A311 * T**n311 * np.exp(-1*E311/R/T) * RKET**1.0 * LIG**1.0  -1*A312 * T**n312 * np.exp(-1*E312/R/T) * PRADIO**1.0 * LIG**1.0  -1*A313 * T**n313 * np.exp(-1*E313/R/T) * RC3H3O**1.0 * LIG**1.0  -1*A314 * T**n314 * np.exp(-1*E314/R/T) * RLIGB**1.0 * LIG**1.0  +1*A314 * T**n314 * np.exp(-1*E314/R/T) * RLIGB**1.0 * LIG**1.0  -1*A315 * T**n315 * np.exp(-1*E315/R/T) * RLIGA**1.0 * LIG**1.0  +1*A315 * T**n315 * np.exp(-1*E315/R/T) * RLIGA**1.0 * LIG**1.0  -1*A316 * T**n316 * np.exp(-1*E316/R/T) * PRADIOM2**1.0 * LIG**1.0  -1*A317 * T**n317 * np.exp(-1*E317/R/T) * RMGUAI**1.0 * LIG**1.0  -1*A318 * T**n318 * np.exp(-1*E318/R/T) * OH**1.0 * LIG**1.0  -1*A319 * T**n319 * np.exp(-1*E319/R/T) * RCH3O**1.0 * LIG**1.0  -1*A320 * T**n320 * np.exp(-1*E320/R/T) * RPHENOL**1.0 * LIG**1.0  -1*A321 * T**n321 * np.exp(-1*E321/R/T) * RADIO**1.0 * LIG**1.0  -1*A322 * T**n322 * np.exp(-1*E322/R/T) * RC3H5O2**1.0 * LIG**1.0  -1*A323 * T**n323 * np.exp(-1*E323/R/T) * PRFET3**1.0 * LIG**1.0  -1*A324 * T**n324 * np.exp(-1*E324/R/T) * RC3H7O2**1.0 * LIG**1.0  -1*A325 * T**n325 * np.exp(-1*E325/R/T) * RADIOM2**1.0 * LIG**1.0  -1*A326 * T**n326 * np.exp(-1*E326/R/T) * PRFET3M2**1.0 * LIG**1.0  -1*A327 * T**n327 * np.exp(-1*E327/R/T) * PRLIGH**1.0 * LIG**1.0  -1*A328 * T**n328 * np.exp(-1*E328/R/T) * RLIGM2B**1.0 * LIG**1.0  -1*A329 * T**n329 * np.exp(-1*E329/R/T) * RLIGM2A**1.0 * LIG**1.0  -1*A330 * T**n330 * np.exp(-1*E330/R/T) * RCH3**1.0 * LIG**1.0  -1*A331 * T**n331 * np.exp(-1*E331/R/T) * PRKETM2**1.0 * LIG**1.0  -1*A332 * T**n332 * np.exp(-1*E332/R/T) * RKET**1.0 * LIG**1.0  -1*A333 * T**n333 * np.exp(-1*E333/R/T) * PRADIO**1.0 * LIG**1.0  -1*A334 * T**n334 * np.exp(-1*E334/R/T) * RC3H3O**1.0 * LIG**1.0  -1*A335 * T**n335 * np.exp(-1*E335/R/T) * RLIGB**1.0 * LIG**1.0  +1*A335 * T**n335 * np.exp(-1*E335/R/T) * RLIGB**1.0 * LIG**1.0  -1*A336 * T**n336 * np.exp(-1*E336/R/T) * RLIGA**1.0 * LIG**1.0  +1*A336 * T**n336 * np.exp(-1*E336/R/T) * RLIGA**1.0 * LIG**1.0  -1*A337 * T**n337 * np.exp(-1*E337/R/T) * PRADIOM2**1.0 * LIG**1.0  -1*A338 * T**n338 * np.exp(-1*E338/R/T) * RMGUAI**1.0 * LIG**1.0  -1*A339 * T**n339 * np.exp(-1*E339/R/T) * OH**1.0 * LIG**1.0  -1*A340 * T**n340 * np.exp(-1*E340/R/T) * RCH3O**1.0 * LIG**1.0  -1*A341 * T**n341 * np.exp(-1*E341/R/T) * RPHENOL**1.0 * LIG**1.0  -1*A342 * T**n342 * np.exp(-1*E342/R/T) * RADIO**1.0 * LIG**1.0  +1*A356 * T**n356 * np.exp(-1*E356/R/T) * RLIGB**1.0 * PFET3**1.0  +1*A357 * T**n357 * np.exp(-1*E357/R/T) * RLIGA**1.0 * PFET3**1.0  +1*A377 * T**n377 * np.exp(-1*E377/R/T) * RLIGB**1.0 * ADIO**1.0  +1*A378 * T**n378 * np.exp(-1*E378/R/T) * RLIGA**1.0 * ADIO**1.0  +1*A398 * T**n398 * np.exp(-1*E398/R/T) * RLIGB**1.0 * KET**1.0  +1*A399 * T**n399 * np.exp(-1*E399/R/T) * RLIGA**1.0 * KET**1.0, 
			+1*A105 * T**n105 * np.exp(-1*E105/R/T) * PLIGC**1.0  -1*A108 * T**n108 * np.exp(-1*E108/R/T) * LIGC**1.0, 
			-1*A1 * T**n1 * np.exp(-1*E1/R/T) * LIGH**1.0  +1*A104 * T**n104 * np.exp(-1*E104/R/T) * PLIGH**1.0  -1*A112 * T**n112 * np.exp(-1*E112/R/T) * RC3H5O2**1.0 * LIGH**1.0  -1*A113 * T**n113 * np.exp(-1*E113/R/T) * PRFET3**1.0 * LIGH**1.0  -1*A114 * T**n114 * np.exp(-1*E114/R/T) * RC3H7O2**1.0 * LIGH**1.0  -1*A115 * T**n115 * np.exp(-1*E115/R/T) * RADIOM2**1.0 * LIGH**1.0  -1*A116 * T**n116 * np.exp(-1*E116/R/T) * PRFET3M2**1.0 * LIGH**1.0  -1*A117 * T**n117 * np.exp(-1*E117/R/T) * PRLIGH**1.0 * LIGH**1.0  -1*A118 * T**n118 * np.exp(-1*E118/R/T) * RLIGM2B**1.0 * LIGH**1.0  -1*A119 * T**n119 * np.exp(-1*E119/R/T) * RLIGM2A**1.0 * LIGH**1.0  -1*A120 * T**n120 * np.exp(-1*E120/R/T) * RCH3**1.0 * LIGH**1.0  -1*A121 * T**n121 * np.exp(-1*E121/R/T) * PRKETM2**1.0 * LIGH**1.0  -1*A122 * T**n122 * np.exp(-1*E122/R/T) * RKET**1.0 * LIGH**1.0  -1*A123 * T**n123 * np.exp(-1*E123/R/T) * PRADIO**1.0 * LIGH**1.0  -1*A124 * T**n124 * np.exp(-1*E124/R/T) * RC3H3O**1.0 * LIGH**1.0  -1*A125 * T**n125 * np.exp(-1*E125/R/T) * RLIGB**1.0 * LIGH**1.0  -1*A126 * T**n126 * np.exp(-1*E126/R/T) * RLIGA**1.0 * LIGH**1.0  -1*A127 * T**n127 * np.exp(-1*E127/R/T) * PRADIOM2**1.0 * LIGH**1.0  -1*A128 * T**n128 * np.exp(-1*E128/R/T) * RMGUAI**1.0 * LIGH**1.0  -1*A129 * T**n129 * np.exp(-1*E129/R/T) * OH**1.0 * LIGH**1.0  -1*A130 * T**n130 * np.exp(-1*E130/R/T) * RCH3O**1.0 * LIGH**1.0  -1*A131 * T**n131 * np.exp(-1*E131/R/T) * RPHENOL**1.0 * LIGH**1.0  -1*A132 * T**n132 * np.exp(-1*E132/R/T) * RADIO**1.0 * LIGH**1.0, 
			-1*A2 * T**n2 * np.exp(-1*E2/R/T) * LIGM2**1.0  +1*A107 * T**n107 * np.exp(-1*E107/R/T) * LIGO**1.0  +1*A118 * T**n118 * np.exp(-1*E118/R/T) * RLIGM2B**1.0 * LIGH**1.0  +1*A119 * T**n119 * np.exp(-1*E119/R/T) * RLIGM2A**1.0 * LIGH**1.0  +1*A139 * T**n139 * np.exp(-1*E139/R/T) * RLIGM2B**1.0 * PLIGH**1.0  +1*A140 * T**n140 * np.exp(-1*E140/R/T) * RLIGM2A**1.0 * PLIGH**1.0  +1*A160 * T**n160 * np.exp(-1*E160/R/T) * RLIGM2B**1.0 * PLIGM2**1.0  +1*A161 * T**n161 * np.exp(-1*E161/R/T) * RLIGM2A**1.0 * PLIGM2**1.0  -1*A175 * T**n175 * np.exp(-1*E175/R/T) * RC3H5O2**1.0 * LIGM2**1.0  -1*A176 * T**n176 * np.exp(-1*E176/R/T) * PRFET3**1.0 * LIGM2**1.0  -1*A177 * T**n177 * np.exp(-1*E177/R/T) * RC3H7O2**1.0 * LIGM2**1.0  -1*A178 * T**n178 * np.exp(-1*E178/R/T) * RADIOM2**1.0 * LIGM2**1.0  -1*A179 * T**n179 * np.exp(-1*E179/R/T) * PRFET3M2**1.0 * LIGM2**1.0  -1*A180 * T**n180 * np.exp(-1*E180/R/T) * PRLIGH**1.0 * LIGM2**1.0  -1*A181 * T**n181 * np.exp(-1*E181/R/T) * RLIGM2B**1.0 * LIGM2**1.0  +1*A181 * T**n181 * np.exp(-1*E181/R/T) * RLIGM2B**1.0 * LIGM2**1.0  -1*A182 * T**n182 * np.exp(-1*E182/R/T) * RLIGM2A**1.0 * LIGM2**1.0  +1*A182 * T**n182 * np.exp(-1*E182/R/T) * RLIGM2A**1.0 * LIGM2**1.0  -1*A183 * T**n183 * np.exp(-1*E183/R/T) * RCH3**1.0 * LIGM2**1.0  -1*A184 * T**n184 * np.exp(-1*E184/R/T) * PRKETM2**1.0 * LIGM2**1.0  -1*A185 * T**n185 * np.exp(-1*E185/R/T) * RKET**1.0 * LIGM2**1.0  -1*A186 * T**n186 * np.exp(-1*E186/R/T) * PRADIO**1.0 * LIGM2**1.0  -1*A187 * T**n187 * np.exp(-1*E187/R/T) * RC3H3O**1.0 * LIGM2**1.0  -1*A188 * T**n188 * np.exp(-1*E188/R/T) * RLIGB**1.0 * LIGM2**1.0  -1*A189 * T**n189 * np.exp(-1*E189/R/T) * RLIGA**1.0 * LIGM2**1.0  -1*A190 * T**n190 * np.exp(-1*E190/R/T) * PRADIOM2**1.0 * LIGM2**1.0  -1*A191 * T**n191 * np.exp(-1*E191/R/T) * RMGUAI**1.0 * LIGM2**1.0  -1*A192 * T**n192 * np.exp(-1*E192/R/T) * OH**1.0 * LIGM2**1.0  -1*A193 * T**n193 * np.exp(-1*E193/R/T) * RCH3O**1.0 * LIGM2**1.0  -1*A194 * T**n194 * np.exp(-1*E194/R/T) * RPHENOL**1.0 * LIGM2**1.0  -1*A195 * T**n195 * np.exp(-1*E195/R/T) * RADIO**1.0 * LIGM2**1.0  -1*A196 * T**n196 * np.exp(-1*E196/R/T) * RC3H5O2**1.0 * LIGM2**1.0  -1*A197 * T**n197 * np.exp(-1*E197/R/T) * PRFET3**1.0 * LIGM2**1.0  -1*A198 * T**n198 * np.exp(-1*E198/R/T) * RC3H7O2**1.0 * LIGM2**1.0  -1*A199 * T**n199 * np.exp(-1*E199/R/T) * RADIOM2**1.0 * LIGM2**1.0  -1*A200 * T**n200 * np.exp(-1*E200/R/T) * PRFET3M2**1.0 * LIGM2**1.0  -1*A201 * T**n201 * np.exp(-1*E201/R/T) * PRLIGH**1.0 * LIGM2**1.0  -1*A202 * T**n202 * np.exp(-1*E202/R/T) * RLIGM2B**1.0 * LIGM2**1.0  +1*A202 * T**n202 * np.exp(-1*E202/R/T) * RLIGM2B**1.0 * LIGM2**1.0  -1*A203 * T**n203 * np.exp(-1*E203/R/T) * RLIGM2A**1.0 * LIGM2**1.0  +1*A203 * T**n203 * np.exp(-1*E203/R/T) * RLIGM2A**1.0 * LIGM2**1.0  -1*A204 * T**n204 * np.exp(-1*E204/R/T) * RCH3**1.0 * LIGM2**1.0  -1*A205 * T**n205 * np.exp(-1*E205/R/T) * PRKETM2**1.0 * LIGM2**1.0  -1*A206 * T**n206 * np.exp(-1*E206/R/T) * RKET**1.0 * LIGM2**1.0  -1*A207 * T**n207 * np.exp(-1*E207/R/T) * PRADIO**1.0 * LIGM2**1.0  -1*A208 * T**n208 * np.exp(-1*E208/R/T) * RC3H3O**1.0 * LIGM2**1.0  -1*A209 * T**n209 * np.exp(-1*E209/R/T) * RLIGB**1.0 * LIGM2**1.0  -1*A210 * T**n210 * np.exp(-1*E210/R/T) * RLIGA**1.0 * LIGM2**1.0  -1*A211 * T**n211 * np.exp(-1*E211/R/T) * PRADIOM2**1.0 * LIGM2**1.0  -1*A212 * T**n212 * np.exp(-1*E212/R/T) * RMGUAI**1.0 * LIGM2**1.0  -1*A213 * T**n213 * np.exp(-1*E213/R/T) * OH**1.0 * LIGM2**1.0  -1*A214 * T**n214 * np.exp(-1*E214/R/T) * RCH3O**1.0 * LIGM2**1.0  -1*A215 * T**n215 * np.exp(-1*E215/R/T) * RPHENOL**1.0 * LIGM2**1.0  -1*A216 * T**n216 * np.exp(-1*E216/R/T) * RADIO**1.0 * LIGM2**1.0  +1*A223 * T**n223 * np.exp(-1*E223/R/T) * RLIGM2B**1.0 * PFET3M2**1.0  +1*A224 * T**n224 * np.exp(-1*E224/R/T) * RLIGM2A**1.0 * PFET3M2**1.0  +1*A244 * T**n244 * np.exp(-1*E244/R/T) * RLIGM2B**1.0 * ADIOM2**1.0  +1*A245 * T**n245 * np.exp(-1*E245/R/T) * RLIGM2A**1.0 * ADIOM2**1.0  +1*A265 * T**n265 * np.exp(-1*E265/R/T) * RLIGM2B**1.0 * KETM2**1.0  +1*A266 * T**n266 * np.exp(-1*E266/R/T) * RLIGM2A**1.0 * KETM2**1.0  +1*A286 * T**n286 * np.exp(-1*E286/R/T) * RLIGM2B**1.0 * C10H2**1.0  +1*A287 * T**n287 * np.exp(-1*E287/R/T) * RLIGM2A**1.0 * C10H2**1.0  +1*A307 * T**n307 * np.exp(-1*E307/R/T) * RLIGM2B**1.0 * LIG**1.0  +1*A308 * T**n308 * np.exp(-1*E308/R/T) * RLIGM2A**1.0 * LIG**1.0  +1*A328 * T**n328 * np.exp(-1*E328/R/T) * RLIGM2B**1.0 * LIG**1.0  +1*A329 * T**n329 * np.exp(-1*E329/R/T) * RLIGM2A**1.0 * LIG**1.0  +1*A349 * T**n349 * np.exp(-1*E349/R/T) * RLIGM2B**1.0 * PFET3**1.0  +1*A350 * T**n350 * np.exp(-1*E350/R/T) * RLIGM2A**1.0 * PFET3**1.0  +1*A370 * T**n370 * np.exp(-1*E370/R/T) * RLIGM2B**1.0 * ADIO**1.0  +1*A371 * T**n371 * np.exp(-1*E371/R/T) * RLIGM2A**1.0 * ADIO**1.0  +1*A391 * T**n391 * np.exp(-1*E391/R/T) * RLIGM2B**1.0 * KET**1.0  +1*A392 * T**n392 * np.exp(-1*E392/R/T) * RLIGM2A**1.0 * KET**1.0, 
			+1*A106 * T**n106 * np.exp(-1*E106/R/T) * PLIGO**1.0  -1*A107 * T**n107 * np.exp(-1*E107/R/T) * LIGO**1.0, 
			-1*A88 * T**n88 * np.exp(-1*E88/R/T) * MGUAI**1.0  +1*A128 * T**n128 * np.exp(-1*E128/R/T) * RMGUAI**1.0 * LIGH**1.0  +1*A149 * T**n149 * np.exp(-1*E149/R/T) * RMGUAI**1.0 * PLIGH**1.0  +1*A170 * T**n170 * np.exp(-1*E170/R/T) * RMGUAI**1.0 * PLIGM2**1.0  +1*A191 * T**n191 * np.exp(-1*E191/R/T) * RMGUAI**1.0 * LIGM2**1.0  +1*A212 * T**n212 * np.exp(-1*E212/R/T) * RMGUAI**1.0 * LIGM2**1.0  +1*A233 * T**n233 * np.exp(-1*E233/R/T) * RMGUAI**1.0 * PFET3M2**1.0  +1*A254 * T**n254 * np.exp(-1*E254/R/T) * RMGUAI**1.0 * ADIOM2**1.0  +1*A275 * T**n275 * np.exp(-1*E275/R/T) * RMGUAI**1.0 * KETM2**1.0  +1*A296 * T**n296 * np.exp(-1*E296/R/T) * RMGUAI**1.0 * C10H2**1.0  +1*A317 * T**n317 * np.exp(-1*E317/R/T) * RMGUAI**1.0 * LIG**1.0  +1*A338 * T**n338 * np.exp(-1*E338/R/T) * RMGUAI**1.0 * LIG**1.0  +1*A359 * T**n359 * np.exp(-1*E359/R/T) * RMGUAI**1.0 * PFET3**1.0  +1*A380 * T**n380 * np.exp(-1*E380/R/T) * RMGUAI**1.0 * ADIO**1.0  +1*A401 * T**n401 * np.exp(-1*E401/R/T) * RMGUAI**1.0 * KET**1.0, 
			+1*A1 * T**n1 * np.exp(-1*E1/R/T) * LIGH**1.0  +1*A9 * T**n9 * np.exp(-1*E9/R/T) * PRKETM2**1.0  +1*A23 * T**n23 * np.exp(-1*E23/R/T) * RADIOM2**1.0  +1*A24 * T**n24 * np.exp(-1*E24/R/T) * RKETM2**1.0  +1*A25 * T**n25 * np.exp(-1*E25/R/T) * RADIO**1.0  +1*A26 * T**n26 * np.exp(-1*E26/R/T) * RKET**1.0  -1*A69 * T**n69 * np.exp(-1*E69/R/T) * OH**1.0 * RCH3**1.0  +1*A98 * T**n98 * np.exp(-1*E98/R/T) * PCH2OH**1.0  +1*A99 * T**n99 * np.exp(-1*E99/R/T) * PCHOHP**1.0  +1*A100 * T**n100 * np.exp(-1*E100/R/T) * PCOHP2**1.0  -1*A129 * T**n129 * np.exp(-1*E129/R/T) * OH**1.0 * LIGH**1.0  -1*A150 * T**n150 * np.exp(-1*E150/R/T) * OH**1.0 * PLIGH**1.0  -1*A171 * T**n171 * np.exp(-1*E171/R/T) * OH**1.0 * PLIGM2**1.0  -1*A192 * T**n192 * np.exp(-1*E192/R/T) * OH**1.0 * LIGM2**1.0  -1*A213 * T**n213 * np.exp(-1*E213/R/T) * OH**1.0 * LIGM2**1.0  -1*A234 * T**n234 * np.exp(-1*E234/R/T) * OH**1.0 * PFET3M2**1.0  -1*A255 * T**n255 * np.exp(-1*E255/R/T) * OH**1.0 * ADIOM2**1.0  -1*A276 * T**n276 * np.exp(-1*E276/R/T) * OH**1.0 * KETM2**1.0  -1*A297 * T**n297 * np.exp(-1*E297/R/T) * OH**1.0 * C10H2**1.0  -1*A318 * T**n318 * np.exp(-1*E318/R/T) * OH**1.0 * LIG**1.0  -1*A339 * T**n339 * np.exp(-1*E339/R/T) * OH**1.0 * LIG**1.0  -1*A360 * T**n360 * np.exp(-1*E360/R/T) * OH**1.0 * PFET3**1.0  -1*A381 * T**n381 * np.exp(-1*E381/R/T) * OH**1.0 * ADIO**1.0  -1*A402 * T**n402 * np.exp(-1*E402/R/T) * OH**1.0 * KET**1.0, 
			-1*A7 * T**n7 * np.exp(-1*E7/R/T) * PADIO**1.0  +1*A123 * T**n123 * np.exp(-1*E123/R/T) * PRADIO**1.0 * LIGH**1.0  +1*A144 * T**n144 * np.exp(-1*E144/R/T) * PRADIO**1.0 * PLIGH**1.0  +1*A165 * T**n165 * np.exp(-1*E165/R/T) * PRADIO**1.0 * PLIGM2**1.0  +1*A186 * T**n186 * np.exp(-1*E186/R/T) * PRADIO**1.0 * LIGM2**1.0  +1*A207 * T**n207 * np.exp(-1*E207/R/T) * PRADIO**1.0 * LIGM2**1.0  +1*A228 * T**n228 * np.exp(-1*E228/R/T) * PRADIO**1.0 * PFET3M2**1.0  +1*A249 * T**n249 * np.exp(-1*E249/R/T) * PRADIO**1.0 * ADIOM2**1.0  +1*A270 * T**n270 * np.exp(-1*E270/R/T) * PRADIO**1.0 * KETM2**1.0  +1*A291 * T**n291 * np.exp(-1*E291/R/T) * PRADIO**1.0 * C10H2**1.0  +1*A312 * T**n312 * np.exp(-1*E312/R/T) * PRADIO**1.0 * LIG**1.0  +1*A333 * T**n333 * np.exp(-1*E333/R/T) * PRADIO**1.0 * LIG**1.0  +1*A354 * T**n354 * np.exp(-1*E354/R/T) * PRADIO**1.0 * PFET3**1.0  +1*A375 * T**n375 * np.exp(-1*E375/R/T) * PRADIO**1.0 * ADIO**1.0  +1*A396 * T**n396 * np.exp(-1*E396/R/T) * PRADIO**1.0 * KET**1.0, 
			-1*A6 * T**n6 * np.exp(-1*E6/R/T) * PADIOM2**1.0  +1*A58 * T**n58 * np.exp(-1*E58/R/T) * RCH3O**1.0 * PRADIOM2**1.0  +1*A71 * T**n71 * np.exp(-1*E71/R/T) * RCH3**1.0 * PRADIOM2**1.0  +1*A127 * T**n127 * np.exp(-1*E127/R/T) * PRADIOM2**1.0 * LIGH**1.0  +1*A148 * T**n148 * np.exp(-1*E148/R/T) * PRADIOM2**1.0 * PLIGH**1.0  +1*A169 * T**n169 * np.exp(-1*E169/R/T) * PRADIOM2**1.0 * PLIGM2**1.0  +1*A190 * T**n190 * np.exp(-1*E190/R/T) * PRADIOM2**1.0 * LIGM2**1.0  +1*A211 * T**n211 * np.exp(-1*E211/R/T) * PRADIOM2**1.0 * LIGM2**1.0  +1*A232 * T**n232 * np.exp(-1*E232/R/T) * PRADIOM2**1.0 * PFET3M2**1.0  +1*A253 * T**n253 * np.exp(-1*E253/R/T) * PRADIOM2**1.0 * ADIOM2**1.0  +1*A274 * T**n274 * np.exp(-1*E274/R/T) * PRADIOM2**1.0 * KETM2**1.0  +1*A295 * T**n295 * np.exp(-1*E295/R/T) * PRADIOM2**1.0 * C10H2**1.0  +1*A316 * T**n316 * np.exp(-1*E316/R/T) * PRADIOM2**1.0 * LIG**1.0  +1*A337 * T**n337 * np.exp(-1*E337/R/T) * PRADIOM2**1.0 * LIG**1.0  +1*A358 * T**n358 * np.exp(-1*E358/R/T) * PRADIOM2**1.0 * PFET3**1.0  +1*A379 * T**n379 * np.exp(-1*E379/R/T) * PRADIOM2**1.0 * ADIO**1.0  +1*A400 * T**n400 * np.exp(-1*E400/R/T) * PRADIOM2**1.0 * KET**1.0, 
			+2*A62 * T**n62 * np.exp(-1*E62/R/T) * RLIGM2A**2.0  +2*A66 * T**n66 * np.exp(-1*E66/R/T) * RC3H7O2**2.0  +2*A67 * T**n67 * np.exp(-1*E67/R/T) * RC3H5O2**2.0  +2*A68 * T**n68 * np.exp(-1*E68/R/T) * RC3H3O**2.0  +1*A73 * T**n73 * np.exp(-1*E73/R/T) * RPHENOX**1.0 * RLIGB**1.0  +2*A76 * T**n76 * np.exp(-1*E76/R/T) * RLIGA**2.0  +2*A79 * T**n79 * np.exp(-1*E79/R/T) * RLIGH**1.0 * RLIGH**1.0  +1*A81 * T**n81 * np.exp(-1*E81/R/T) * RPHENOX**1.0 * RC3H3O**1.0  -1*A97 * T**n97 * np.exp(-1*E97/R/T) * PC2H2**1.0  +0.5*A99 * T**n99 * np.exp(-1*E99/R/T) * PCHOHP**1.0, 
			+1*A58 * T**n58 * np.exp(-1*E58/R/T) * RCH3O**1.0 * PRADIOM2**1.0  +1*A59 * T**n59 * np.exp(-1*E59/R/T) * RCH3O**1.0 * PRKETM2**1.0  +2*A60 * T**n60 * np.exp(-1*E60/R/T) * RADIOM2**2.0  +2*A61 * T**n61 * np.exp(-1*E61/R/T) * RLIGM2B**2.0  +2*A64 * T**n64 * np.exp(-1*E64/R/T) * RKETM2**2.0  +2*A74 * T**n74 * np.exp(-1*E74/R/T) * RADIO**2.0  +2*A75 * T**n75 * np.exp(-1*E75/R/T) * RLIGB**1.0 * RLIGB**1.0  +2*A77 * T**n77 * np.exp(-1*E77/R/T) * RKET**2.0  -1*A98 * T**n98 * np.exp(-1*E98/R/T) * PCH2OH**1.0  +2*A109 * T**n109 * np.exp(-1*E109/R/T) * PRADIO**2.0  +2*A110 * T**n110 * np.exp(-1*E110/R/T) * PRADIOM2**2.0, 
			+2*A79 * T**n79 * np.exp(-1*E79/R/T) * RLIGH**1.0 * RLIGH**1.0  +1*A98 * T**n98 * np.exp(-1*E98/R/T) * PCH2OH**1.0  -1*A102 * T**n102 * np.exp(-1*E102/R/T) * PCH2P**1.0, 
			+1*A54 * T**n54 * np.exp(-1*E54/R/T) * RPHENOXM2**1.0 * RCH3**1.0  +1*A55 * T**n55 * np.exp(-1*E55/R/T) * RPHENOX**1.0 * RCH3**1.0  +1*A71 * T**n71 * np.exp(-1*E71/R/T) * RCH3**1.0 * PRADIOM2**1.0  +1*A72 * T**n72 * np.exp(-1*E72/R/T) * RCH3**1.0 * PRKETM2**1.0  +2*A79 * T**n79 * np.exp(-1*E79/R/T) * RLIGH**1.0 * RLIGH**1.0  -1*A101 * T**n101 * np.exp(-1*E101/R/T) * PCH3**1.0, 
			+2*A65 * T**n65 * np.exp(-1*E65/R/T) * PRFET3M2**1.0 * PRFET3M2**1.0  +2*A78 * T**n78 * np.exp(-1*E78/R/T) * PRFET3**1.0 * PRFET3**1.0  -1*A111 * T**n111 * np.exp(-1*E111/R/T) * PCHO**1.0, 
			+2*A60 * T**n60 * np.exp(-1*E60/R/T) * RADIOM2**2.0  +2*A61 * T**n61 * np.exp(-1*E61/R/T) * RLIGM2B**2.0  +2*A65 * T**n65 * np.exp(-1*E65/R/T) * PRFET3M2**1.0 * PRFET3M2**1.0  +2*A66 * T**n66 * np.exp(-1*E66/R/T) * RC3H7O2**2.0  +2*A74 * T**n74 * np.exp(-1*E74/R/T) * RADIO**2.0  +2*A75 * T**n75 * np.exp(-1*E75/R/T) * RLIGB**1.0 * RLIGB**1.0  +2*A78 * T**n78 * np.exp(-1*E78/R/T) * PRFET3**1.0 * PRFET3**1.0  -1*A99 * T**n99 * np.exp(-1*E99/R/T) * PCHOHP**1.0  +2*A109 * T**n109 * np.exp(-1*E109/R/T) * PRADIO**2.0  +2*A110 * T**n110 * np.exp(-1*E110/R/T) * PRADIOM2**2.0, 
			+2*A60 * T**n60 * np.exp(-1*E60/R/T) * RADIOM2**2.0  +2*A64 * T**n64 * np.exp(-1*E64/R/T) * RKETM2**2.0  +2*A65 * T**n65 * np.exp(-1*E65/R/T) * PRFET3M2**1.0 * PRFET3M2**1.0  +2*A74 * T**n74 * np.exp(-1*E74/R/T) * RADIO**2.0  +2*A77 * T**n77 * np.exp(-1*E77/R/T) * RKET**2.0  +2*A78 * T**n78 * np.exp(-1*E78/R/T) * PRFET3**1.0 * PRFET3**1.0  +2*A79 * T**n79 * np.exp(-1*E79/R/T) * RLIGH**1.0 * RLIGH**1.0  +1*A102 * T**n102 * np.exp(-1*E102/R/T) * PCH2P**1.0  -1*A103 * T**n103 * np.exp(-1*E103/R/T) * PCHP2**1.0  +2*A109 * T**n109 * np.exp(-1*E109/R/T) * PRADIO**2.0  +2*A110 * T**n110 * np.exp(-1*E110/R/T) * PRADIOM2**2.0, 
			+2*A32 * T**n32 * np.exp(-1*E32/R/T) * ADIOM2**1.0 * RPHENOXM2**1.0  +2*A33 * T**n33 * np.exp(-1*E33/R/T) * KETM2**1.0 * RPHENOXM2**1.0  +2*A34 * T**n34 * np.exp(-1*E34/R/T) * KETDM2**1.0 * RPHENOXM2**1.0  +2*A35 * T**n35 * np.exp(-1*E35/R/T) * SYNAPYL**1.0 * RPHENOXM2**1.0  +1*A36 * T**n36 * np.exp(-1*E36/R/T) * ADIOM2**1.0 * RPHENOX**1.0  +1*A38 * T**n38 * np.exp(-1*E38/R/T) * KETDM2**1.0 * RPHENOX**1.0  +1*A39 * T**n39 * np.exp(-1*E39/R/T) * SYNAPYL**1.0 * RPHENOX**1.0  +1*A44 * T**n44 * np.exp(-1*E44/R/T) * ADIO**1.0 * RPHENOXM2**1.0  +1*A45 * T**n45 * np.exp(-1*E45/R/T) * KET**1.0 * RPHENOXM2**1.0  +1*A46 * T**n46 * np.exp(-1*E46/R/T) * KETD**1.0 * RPHENOXM2**1.0  +1*A47 * T**n47 * np.exp(-1*E47/R/T) * COUMARYL**1.0 * RPHENOXM2**1.0  +1*A48 * T**n48 * np.exp(-1*E48/R/T) * C10H2M4**1.0 * RPHENOXM2**1.0  +1*A49 * T**n49 * np.exp(-1*E49/R/T) * C10H2M2**1.0 * RPHENOXM2**1.0  +1*A54 * T**n54 * np.exp(-1*E54/R/T) * RPHENOXM2**1.0 * RCH3**1.0  +1*A55 * T**n55 * np.exp(-1*E55/R/T) * RPHENOX**1.0 * RCH3**1.0  +2*A60 * T**n60 * np.exp(-1*E60/R/T) * RADIOM2**2.0  +4*A61 * T**n61 * np.exp(-1*E61/R/T) * RLIGM2B**2.0  +4*A62 * T**n62 * np.exp(-1*E62/R/T) * RLIGM2A**2.0  +2*A63 * T**n63 * np.exp(-1*E63/R/T) * RMGUAI**2.0  +4*A64 * T**n64 * np.exp(-1*E64/R/T) * RKETM2**2.0  +2*A65 * T**n65 * np.exp(-1*E65/R/T) * PRFET3M2**1.0 * PRFET3M2**1.0  +2*A67 * T**n67 * np.exp(-1*E67/R/T) * RC3H5O2**2.0  +2*A68 * T**n68 * np.exp(-1*E68/R/T) * RC3H3O**2.0  +1*A74 * T**n74 * np.exp(-1*E74/R/T) * RADIO**2.0  +3*A77 * T**n77 * np.exp(-1*E77/R/T) * RKET**2.0  +4*A79 * T**n79 * np.exp(-1*E79/R/T) * RLIGH**1.0 * RLIGH**1.0  +1*A81 * T**n81 * np.exp(-1*E81/R/T) * RPHENOX**1.0 * RC3H3O**1.0  -1*A95 * T**n95 * np.exp(-1*E95/R/T) * PCOH**1.0  +1*A109 * T**n109 * np.exp(-1*E109/R/T) * PRADIO**2.0  +2*A110 * T**n110 * np.exp(-1*E110/R/T) * PRADIOM2**2.0, 
			+2*A62 * T**n62 * np.exp(-1*E62/R/T) * RLIGM2A**2.0  +1*A73 * T**n73 * np.exp(-1*E73/R/T) * RPHENOX**1.0 * RLIGB**1.0  +2*A76 * T**n76 * np.exp(-1*E76/R/T) * RLIGA**2.0  +2*A79 * T**n79 * np.exp(-1*E79/R/T) * RLIGH**1.0 * RLIGH**1.0  -1*A100 * T**n100 * np.exp(-1*E100/R/T) * PCOHP2**1.0, 
			+1*A36 * T**n36 * np.exp(-1*E36/R/T) * ADIOM2**1.0 * RPHENOX**1.0  +2*A37 * T**n37 * np.exp(-1*E37/R/T) * KETM2**1.0 * RPHENOX**1.0  +1*A38 * T**n38 * np.exp(-1*E38/R/T) * KETDM2**1.0 * RPHENOX**1.0  +1*A39 * T**n39 * np.exp(-1*E39/R/T) * SYNAPYL**1.0 * RPHENOX**1.0  +2*A40 * T**n40 * np.exp(-1*E40/R/T) * ADIO**1.0 * RPHENOX**1.0  +2*A41 * T**n41 * np.exp(-1*E41/R/T) * KET**1.0 * RPHENOX**1.0  +2*A42 * T**n42 * np.exp(-1*E42/R/T) * KETD**1.0 * RPHENOX**1.0  +2*A43 * T**n43 * np.exp(-1*E43/R/T) * COUMARYL**1.0 * RPHENOX**1.0  +1*A44 * T**n44 * np.exp(-1*E44/R/T) * ADIO**1.0 * RPHENOXM2**1.0  +1*A45 * T**n45 * np.exp(-1*E45/R/T) * KET**1.0 * RPHENOXM2**1.0  +1*A46 * T**n46 * np.exp(-1*E46/R/T) * KETD**1.0 * RPHENOXM2**1.0  +1*A47 * T**n47 * np.exp(-1*E47/R/T) * COUMARYL**1.0 * RPHENOXM2**1.0  +1*A50 * T**n50 * np.exp(-1*E50/R/T) * C10H2M4**1.0 * RPHENOX**1.0  +1*A51 * T**n51 * np.exp(-1*E51/R/T) * C10H2M2**1.0 * RPHENOX**1.0  +3*A73 * T**n73 * np.exp(-1*E73/R/T) * RPHENOX**1.0 * RLIGB**1.0  +1*A74 * T**n74 * np.exp(-1*E74/R/T) * RADIO**2.0  +1*A75 * T**n75 * np.exp(-1*E75/R/T) * RLIGB**1.0 * RLIGB**1.0  +3*A75 * T**n75 * np.exp(-1*E75/R/T) * RLIGB**1.0 * RLIGB**1.0  +4*A76 * T**n76 * np.exp(-1*E76/R/T) * RLIGA**2.0  +1*A77 * T**n77 * np.exp(-1*E77/R/T) * RKET**2.0  +2*A78 * T**n78 * np.exp(-1*E78/R/T) * PRFET3**1.0 * PRFET3**1.0  +2*A80 * T**n80 * np.exp(-1*E80/R/T) * RPHENOX**1.0 * RPHENOL**1.0  +1*A81 * T**n81 * np.exp(-1*E81/R/T) * RPHENOX**1.0 * RC3H3O**1.0  +1*A82 * T**n82 * np.exp(-1*E82/R/T) * RPHENOX**1.0 * CHAR**1.0  +1*A83 * T**n83 * np.exp(-1*E83/R/T) * RPHENOXM2**1.0 * CHAR**1.0  -1*A94 * T**n94 * np.exp(-1*E94/R/T) * PCOS**1.0  +1*A109 * T**n109 * np.exp(-1*E109/R/T) * PRADIO**2.0, 
			+1*A21 * T**n21 * np.exp(-1*E21/R/T) * RLIGB**1.0  +1*A113 * T**n113 * np.exp(-1*E113/R/T) * PRFET3**1.0 * LIGH**1.0  +1*A134 * T**n134 * np.exp(-1*E134/R/T) * PRFET3**1.0 * PLIGH**1.0  +1*A155 * T**n155 * np.exp(-1*E155/R/T) * PRFET3**1.0 * PLIGM2**1.0  +1*A176 * T**n176 * np.exp(-1*E176/R/T) * PRFET3**1.0 * LIGM2**1.0  +1*A197 * T**n197 * np.exp(-1*E197/R/T) * PRFET3**1.0 * LIGM2**1.0  +1*A218 * T**n218 * np.exp(-1*E218/R/T) * PRFET3**1.0 * PFET3M2**1.0  +1*A239 * T**n239 * np.exp(-1*E239/R/T) * PRFET3**1.0 * ADIOM2**1.0  +1*A260 * T**n260 * np.exp(-1*E260/R/T) * PRFET3**1.0 * KETM2**1.0  +1*A281 * T**n281 * np.exp(-1*E281/R/T) * PRFET3**1.0 * C10H2**1.0  +1*A302 * T**n302 * np.exp(-1*E302/R/T) * PRFET3**1.0 * LIG**1.0  +1*A323 * T**n323 * np.exp(-1*E323/R/T) * PRFET3**1.0 * LIG**1.0  -1*A343 * T**n343 * np.exp(-1*E343/R/T) * RC3H5O2**1.0 * PFET3**1.0  -1*A344 * T**n344 * np.exp(-1*E344/R/T) * PRFET3**1.0 * PFET3**1.0  +1*A344 * T**n344 * np.exp(-1*E344/R/T) * PRFET3**1.0 * PFET3**1.0  -1*A345 * T**n345 * np.exp(-1*E345/R/T) * RC3H7O2**1.0 * PFET3**1.0  -1*A346 * T**n346 * np.exp(-1*E346/R/T) * RADIOM2**1.0 * PFET3**1.0  -1*A347 * T**n347 * np.exp(-1*E347/R/T) * PRFET3M2**1.0 * PFET3**1.0  -1*A348 * T**n348 * np.exp(-1*E348/R/T) * PRLIGH**1.0 * PFET3**1.0  -1*A349 * T**n349 * np.exp(-1*E349/R/T) * RLIGM2B**1.0 * PFET3**1.0  -1*A350 * T**n350 * np.exp(-1*E350/R/T) * RLIGM2A**1.0 * PFET3**1.0  -1*A351 * T**n351 * np.exp(-1*E351/R/T) * RCH3**1.0 * PFET3**1.0  -1*A352 * T**n352 * np.exp(-1*E352/R/T) * PRKETM2**1.0 * PFET3**1.0  -1*A353 * T**n353 * np.exp(-1*E353/R/T) * RKET**1.0 * PFET3**1.0  -1*A354 * T**n354 * np.exp(-1*E354/R/T) * PRADIO**1.0 * PFET3**1.0  -1*A355 * T**n355 * np.exp(-1*E355/R/T) * RC3H3O**1.0 * PFET3**1.0  -1*A356 * T**n356 * np.exp(-1*E356/R/T) * RLIGB**1.0 * PFET3**1.0  -1*A357 * T**n357 * np.exp(-1*E357/R/T) * RLIGA**1.0 * PFET3**1.0  -1*A358 * T**n358 * np.exp(-1*E358/R/T) * PRADIOM2**1.0 * PFET3**1.0  -1*A359 * T**n359 * np.exp(-1*E359/R/T) * RMGUAI**1.0 * PFET3**1.0  -1*A360 * T**n360 * np.exp(-1*E360/R/T) * OH**1.0 * PFET3**1.0  -1*A361 * T**n361 * np.exp(-1*E361/R/T) * RCH3O**1.0 * PFET3**1.0  -1*A362 * T**n362 * np.exp(-1*E362/R/T) * RPHENOL**1.0 * PFET3**1.0  -1*A363 * T**n363 * np.exp(-1*E363/R/T) * RADIO**1.0 * PFET3**1.0  +1*A365 * T**n365 * np.exp(-1*E365/R/T) * PRFET3**1.0 * ADIO**1.0  +1*A386 * T**n386 * np.exp(-1*E386/R/T) * PRFET3**1.0 * KET**1.0, 
			+1*A17 * T**n17 * np.exp(-1*E17/R/T) * RLIGM2B**1.0  +1*A116 * T**n116 * np.exp(-1*E116/R/T) * PRFET3M2**1.0 * LIGH**1.0  +1*A137 * T**n137 * np.exp(-1*E137/R/T) * PRFET3M2**1.0 * PLIGH**1.0  +1*A158 * T**n158 * np.exp(-1*E158/R/T) * PRFET3M2**1.0 * PLIGM2**1.0  +1*A179 * T**n179 * np.exp(-1*E179/R/T) * PRFET3M2**1.0 * LIGM2**1.0  +1*A200 * T**n200 * np.exp(-1*E200/R/T) * PRFET3M2**1.0 * LIGM2**1.0  -1*A217 * T**n217 * np.exp(-1*E217/R/T) * RC3H5O2**1.0 * PFET3M2**1.0  -1*A218 * T**n218 * np.exp(-1*E218/R/T) * PRFET3**1.0 * PFET3M2**1.0  -1*A219 * T**n219 * np.exp(-1*E219/R/T) * RC3H7O2**1.0 * PFET3M2**1.0  -1*A220 * T**n220 * np.exp(-1*E220/R/T) * RADIOM2**1.0 * PFET3M2**1.0  -1*A221 * T**n221 * np.exp(-1*E221/R/T) * PRFET3M2**1.0 * PFET3M2**1.0  +1*A221 * T**n221 * np.exp(-1*E221/R/T) * PRFET3M2**1.0 * PFET3M2**1.0  -1*A222 * T**n222 * np.exp(-1*E222/R/T) * PRLIGH**1.0 * PFET3M2**1.0  -1*A223 * T**n223 * np.exp(-1*E223/R/T) * RLIGM2B**1.0 * PFET3M2**1.0  -1*A224 * T**n224 * np.exp(-1*E224/R/T) * RLIGM2A**1.0 * PFET3M2**1.0  -1*A225 * T**n225 * np.exp(-1*E225/R/T) * RCH3**1.0 * PFET3M2**1.0  -1*A226 * T**n226 * np.exp(-1*E226/R/T) * PRKETM2**1.0 * PFET3M2**1.0  -1*A227 * T**n227 * np.exp(-1*E227/R/T) * RKET**1.0 * PFET3M2**1.0  -1*A228 * T**n228 * np.exp(-1*E228/R/T) * PRADIO**1.0 * PFET3M2**1.0  -1*A229 * T**n229 * np.exp(-1*E229/R/T) * RC3H3O**1.0 * PFET3M2**1.0  -1*A230 * T**n230 * np.exp(-1*E230/R/T) * RLIGB**1.0 * PFET3M2**1.0  -1*A231 * T**n231 * np.exp(-1*E231/R/T) * RLIGA**1.0 * PFET3M2**1.0  -1*A232 * T**n232 * np.exp(-1*E232/R/T) * PRADIOM2**1.0 * PFET3M2**1.0  -1*A233 * T**n233 * np.exp(-1*E233/R/T) * RMGUAI**1.0 * PFET3M2**1.0  -1*A234 * T**n234 * np.exp(-1*E234/R/T) * OH**1.0 * PFET3M2**1.0  -1*A235 * T**n235 * np.exp(-1*E235/R/T) * RCH3O**1.0 * PFET3M2**1.0  -1*A236 * T**n236 * np.exp(-1*E236/R/T) * RPHENOL**1.0 * PFET3M2**1.0  -1*A237 * T**n237 * np.exp(-1*E237/R/T) * RADIO**1.0 * PFET3M2**1.0  +1*A242 * T**n242 * np.exp(-1*E242/R/T) * PRFET3M2**1.0 * ADIOM2**1.0  +1*A263 * T**n263 * np.exp(-1*E263/R/T) * PRFET3M2**1.0 * KETM2**1.0  +1*A284 * T**n284 * np.exp(-1*E284/R/T) * PRFET3M2**1.0 * C10H2**1.0  +1*A305 * T**n305 * np.exp(-1*E305/R/T) * PRFET3M2**1.0 * LIG**1.0  +1*A326 * T**n326 * np.exp(-1*E326/R/T) * PRFET3M2**1.0 * LIG**1.0  +1*A347 * T**n347 * np.exp(-1*E347/R/T) * PRFET3M2**1.0 * PFET3**1.0  +1*A368 * T**n368 * np.exp(-1*E368/R/T) * PRFET3M2**1.0 * ADIO**1.0  +1*A389 * T**n389 * np.exp(-1*E389/R/T) * PRFET3M2**1.0 * KET**1.0, 
			+1*A32 * T**n32 * np.exp(-1*E32/R/T) * ADIOM2**1.0 * RPHENOXM2**1.0  +1*A33 * T**n33 * np.exp(-1*E33/R/T) * KETM2**1.0 * RPHENOXM2**1.0  +1*A34 * T**n34 * np.exp(-1*E34/R/T) * KETDM2**1.0 * RPHENOXM2**1.0  +1.5*A35 * T**n35 * np.exp(-1*E35/R/T) * SYNAPYL**1.0 * RPHENOXM2**1.0  +1*A36 * T**n36 * np.exp(-1*E36/R/T) * ADIOM2**1.0 * RPHENOX**1.0  +2*A37 * T**n37 * np.exp(-1*E37/R/T) * KETM2**1.0 * RPHENOX**1.0  +2*A38 * T**n38 * np.exp(-1*E38/R/T) * KETDM2**1.0 * RPHENOX**1.0  +2*A39 * T**n39 * np.exp(-1*E39/R/T) * SYNAPYL**1.0 * RPHENOX**1.0  +2*A40 * T**n40 * np.exp(-1*E40/R/T) * ADIO**1.0 * RPHENOX**1.0  +2*A41 * T**n41 * np.exp(-1*E41/R/T) * KET**1.0 * RPHENOX**1.0  +2*A42 * T**n42 * np.exp(-1*E42/R/T) * KETD**1.0 * RPHENOX**1.0  +2*A43 * T**n43 * np.exp(-1*E43/R/T) * COUMARYL**1.0 * RPHENOX**1.0  +1*A44 * T**n44 * np.exp(-1*E44/R/T) * ADIO**1.0 * RPHENOXM2**1.0  +1*A45 * T**n45 * np.exp(-1*E45/R/T) * KET**1.0 * RPHENOXM2**1.0  +1*A46 * T**n46 * np.exp(-1*E46/R/T) * KETD**1.0 * RPHENOXM2**1.0  +2*A47 * T**n47 * np.exp(-1*E47/R/T) * COUMARYL**1.0 * RPHENOXM2**1.0  +0.5*A54 * T**n54 * np.exp(-1*E54/R/T) * RPHENOXM2**1.0 * RCH3**1.0  +1.5*A55 * T**n55 * np.exp(-1*E55/R/T) * RPHENOX**1.0 * RCH3**1.0  +2*A60 * T**n60 * np.exp(-1*E60/R/T) * RADIOM2**2.0  +3*A61 * T**n61 * np.exp(-1*E61/R/T) * RLIGM2B**2.0  +3*A62 * T**n62 * np.exp(-1*E62/R/T) * RLIGM2A**2.0  +2*A63 * T**n63 * np.exp(-1*E63/R/T) * RMGUAI**2.0  +2*A64 * T**n64 * np.exp(-1*E64/R/T) * RKETM2**2.0  +1*A65 * T**n65 * np.exp(-1*E65/R/T) * PRFET3M2**1.0 * PRFET3M2**1.0  +1*A66 * T**n66 * np.exp(-1*E66/R/T) * RC3H7O2**2.0  +1*A67 * T**n67 * np.exp(-1*E67/R/T) * RC3H5O2**2.0  +1*A68 * T**n68 * np.exp(-1*E68/R/T) * RC3H3O**2.0  +3*A73 * T**n73 * np.exp(-1*E73/R/T) * RPHENOX**1.0 * RLIGB**1.0  +3*A74 * T**n74 * np.exp(-1*E74/R/T) * RADIO**2.0  +5*A75 * T**n75 * np.exp(-1*E75/R/T) * RLIGB**1.0 * RLIGB**1.0  +5*A76 * T**n76 * np.exp(-1*E76/R/T) * RLIGA**2.0  +2*A77 * T**n77 * np.exp(-1*E77/R/T) * RKET**2.0  +2*A78 * T**n78 * np.exp(-1*E78/R/T) * PRFET3**1.0 * PRFET3**1.0  +2*A80 * T**n80 * np.exp(-1*E80/R/T) * RPHENOX**1.0 * RPHENOL**1.0  +1*A81 * T**n81 * np.exp(-1*E81/R/T) * RPHENOX**1.0 * RC3H3O**1.0  -1*A96 * T**n96 * np.exp(-1*E96/R/T) * PH2**1.0  +2*A109 * T**n109 * np.exp(-1*E109/R/T) * PRADIO**2.0  +1*A110 * T**n110 * np.exp(-1*E110/R/T) * PRADIOM2**2.0, 
			-1*A93 * T**n93 * np.exp(-1*E93/R/T) * PHENOL**1.0  +1*A131 * T**n131 * np.exp(-1*E131/R/T) * RPHENOL**1.0 * LIGH**1.0  +1*A152 * T**n152 * np.exp(-1*E152/R/T) * RPHENOL**1.0 * PLIGH**1.0  +1*A173 * T**n173 * np.exp(-1*E173/R/T) * RPHENOL**1.0 * PLIGM2**1.0  +1*A194 * T**n194 * np.exp(-1*E194/R/T) * RPHENOL**1.0 * LIGM2**1.0  +1*A215 * T**n215 * np.exp(-1*E215/R/T) * RPHENOL**1.0 * LIGM2**1.0  +1*A236 * T**n236 * np.exp(-1*E236/R/T) * RPHENOL**1.0 * PFET3M2**1.0  +1*A257 * T**n257 * np.exp(-1*E257/R/T) * RPHENOL**1.0 * ADIOM2**1.0  +1*A278 * T**n278 * np.exp(-1*E278/R/T) * RPHENOL**1.0 * KETM2**1.0  +1*A299 * T**n299 * np.exp(-1*E299/R/T) * RPHENOL**1.0 * C10H2**1.0  +1*A320 * T**n320 * np.exp(-1*E320/R/T) * RPHENOL**1.0 * LIG**1.0  +1*A341 * T**n341 * np.exp(-1*E341/R/T) * RPHENOL**1.0 * LIG**1.0  +1*A362 * T**n362 * np.exp(-1*E362/R/T) * RPHENOL**1.0 * PFET3**1.0  +1*A383 * T**n383 * np.exp(-1*E383/R/T) * RPHENOL**1.0 * ADIO**1.0  +1*A404 * T**n404 * np.exp(-1*E404/R/T) * RPHENOL**1.0 * KET**1.0, 
			-1*A8 * T**n8 * np.exp(-1*E8/R/T) * PKETM2**1.0  +1*A16 * T**n16 * np.exp(-1*E16/R/T) * PRLIGM2A**1.0  +1*A59 * T**n59 * np.exp(-1*E59/R/T) * RCH3O**1.0 * PRKETM2**1.0  +1*A72 * T**n72 * np.exp(-1*E72/R/T) * RCH3**1.0 * PRKETM2**1.0  +1*A121 * T**n121 * np.exp(-1*E121/R/T) * PRKETM2**1.0 * LIGH**1.0  +1*A142 * T**n142 * np.exp(-1*E142/R/T) * PRKETM2**1.0 * PLIGH**1.0  +1*A163 * T**n163 * np.exp(-1*E163/R/T) * PRKETM2**1.0 * PLIGM2**1.0  +1*A184 * T**n184 * np.exp(-1*E184/R/T) * PRKETM2**1.0 * LIGM2**1.0  +1*A205 * T**n205 * np.exp(-1*E205/R/T) * PRKETM2**1.0 * LIGM2**1.0  +1*A226 * T**n226 * np.exp(-1*E226/R/T) * PRKETM2**1.0 * PFET3M2**1.0  +1*A247 * T**n247 * np.exp(-1*E247/R/T) * PRKETM2**1.0 * ADIOM2**1.0  +1*A268 * T**n268 * np.exp(-1*E268/R/T) * PRKETM2**1.0 * KETM2**1.0  +1*A289 * T**n289 * np.exp(-1*E289/R/T) * PRKETM2**1.0 * C10H2**1.0  +1*A310 * T**n310 * np.exp(-1*E310/R/T) * PRKETM2**1.0 * LIG**1.0  +1*A331 * T**n331 * np.exp(-1*E331/R/T) * PRKETM2**1.0 * LIG**1.0  +1*A352 * T**n352 * np.exp(-1*E352/R/T) * PRKETM2**1.0 * PFET3**1.0  +1*A373 * T**n373 * np.exp(-1*E373/R/T) * PRKETM2**1.0 * ADIO**1.0  +1*A394 * T**n394 * np.exp(-1*E394/R/T) * PRKETM2**1.0 * KET**1.0, 
			-1*A5 * T**n5 * np.exp(-1*E5/R/T) * PLIG**1.0  +1*A30 * T**n30 * np.exp(-1*E30/R/T) * PLIGC**1.0, 
			-1*A30 * T**n30 * np.exp(-1*E30/R/T) * PLIGC**1.0  -1*A105 * T**n105 * np.exp(-1*E105/R/T) * PLIGC**1.0, 
			-1*A0 * T**n0 * np.exp(-1*E0/R/T) * PLIGH**1.0  -1*A104 * T**n104 * np.exp(-1*E104/R/T) * PLIGH**1.0  +1*A117 * T**n117 * np.exp(-1*E117/R/T) * PRLIGH**1.0 * LIGH**1.0  -1*A133 * T**n133 * np.exp(-1*E133/R/T) * RC3H5O2**1.0 * PLIGH**1.0  -1*A134 * T**n134 * np.exp(-1*E134/R/T) * PRFET3**1.0 * PLIGH**1.0  -1*A135 * T**n135 * np.exp(-1*E135/R/T) * RC3H7O2**1.0 * PLIGH**1.0  -1*A136 * T**n136 * np.exp(-1*E136/R/T) * RADIOM2**1.0 * PLIGH**1.0  -1*A137 * T**n137 * np.exp(-1*E137/R/T) * PRFET3M2**1.0 * PLIGH**1.0  -1*A138 * T**n138 * np.exp(-1*E138/R/T) * PRLIGH**1.0 * PLIGH**1.0  +1*A138 * T**n138 * np.exp(-1*E138/R/T) * PRLIGH**1.0 * PLIGH**1.0  -1*A139 * T**n139 * np.exp(-1*E139/R/T) * RLIGM2B**1.0 * PLIGH**1.0  -1*A140 * T**n140 * np.exp(-1*E140/R/T) * RLIGM2A**1.0 * PLIGH**1.0  -1*A141 * T**n141 * np.exp(-1*E141/R/T) * RCH3**1.0 * PLIGH**1.0  -1*A142 * T**n142 * np.exp(-1*E142/R/T) * PRKETM2**1.0 * PLIGH**1.0  -1*A143 * T**n143 * np.exp(-1*E143/R/T) * RKET**1.0 * PLIGH**1.0  -1*A144 * T**n144 * np.exp(-1*E144/R/T) * PRADIO**1.0 * PLIGH**1.0  -1*A145 * T**n145 * np.exp(-1*E145/R/T) * RC3H3O**1.0 * PLIGH**1.0  -1*A146 * T**n146 * np.exp(-1*E146/R/T) * RLIGB**1.0 * PLIGH**1.0  -1*A147 * T**n147 * np.exp(-1*E147/R/T) * RLIGA**1.0 * PLIGH**1.0  -1*A148 * T**n148 * np.exp(-1*E148/R/T) * PRADIOM2**1.0 * PLIGH**1.0  -1*A149 * T**n149 * np.exp(-1*E149/R/T) * RMGUAI**1.0 * PLIGH**1.0  -1*A150 * T**n150 * np.exp(-1*E150/R/T) * OH**1.0 * PLIGH**1.0  -1*A151 * T**n151 * np.exp(-1*E151/R/T) * RCH3O**1.0 * PLIGH**1.0  -1*A152 * T**n152 * np.exp(-1*E152/R/T) * RPHENOL**1.0 * PLIGH**1.0  -1*A153 * T**n153 * np.exp(-1*E153/R/T) * RADIO**1.0 * PLIGH**1.0  +1*A159 * T**n159 * np.exp(-1*E159/R/T) * PRLIGH**1.0 * PLIGM2**1.0  +1*A180 * T**n180 * np.exp(-1*E180/R/T) * PRLIGH**1.0 * LIGM2**1.0  +1*A201 * T**n201 * np.exp(-1*E201/R/T) * PRLIGH**1.0 * LIGM2**1.0  +1*A222 * T**n222 * np.exp(-1*E222/R/T) * PRLIGH**1.0 * PFET3M2**1.0  +1*A243 * T**n243 * np.exp(-1*E243/R/T) * PRLIGH**1.0 * ADIOM2**1.0  +1*A264 * T**n264 * np.exp(-1*E264/R/T) * PRLIGH**1.0 * KETM2**1.0  +1*A285 * T**n285 * np.exp(-1*E285/R/T) * PRLIGH**1.0 * C10H2**1.0  +1*A306 * T**n306 * np.exp(-1*E306/R/T) * PRLIGH**1.0 * LIG**1.0  +1*A327 * T**n327 * np.exp(-1*E327/R/T) * PRLIGH**1.0 * LIG**1.0  +1*A348 * T**n348 * np.exp(-1*E348/R/T) * PRLIGH**1.0 * PFET3**1.0  +1*A369 * T**n369 * np.exp(-1*E369/R/T) * PRLIGH**1.0 * ADIO**1.0  +1*A390 * T**n390 * np.exp(-1*E390/R/T) * PRLIGH**1.0 * KET**1.0, 
			-1*A3 * T**n3 * np.exp(-1*E3/R/T) * PLIGM2**1.0  +1*A31 * T**n31 * np.exp(-1*E31/R/T) * PLIGO**1.0  -1*A154 * T**n154 * np.exp(-1*E154/R/T) * RC3H5O2**1.0 * PLIGM2**1.0  -1*A155 * T**n155 * np.exp(-1*E155/R/T) * PRFET3**1.0 * PLIGM2**1.0  -1*A156 * T**n156 * np.exp(-1*E156/R/T) * RC3H7O2**1.0 * PLIGM2**1.0  -1*A157 * T**n157 * np.exp(-1*E157/R/T) * RADIOM2**1.0 * PLIGM2**1.0  -1*A158 * T**n158 * np.exp(-1*E158/R/T) * PRFET3M2**1.0 * PLIGM2**1.0  -1*A159 * T**n159 * np.exp(-1*E159/R/T) * PRLIGH**1.0 * PLIGM2**1.0  -1*A160 * T**n160 * np.exp(-1*E160/R/T) * RLIGM2B**1.0 * PLIGM2**1.0  -1*A161 * T**n161 * np.exp(-1*E161/R/T) * RLIGM2A**1.0 * PLIGM2**1.0  -1*A162 * T**n162 * np.exp(-1*E162/R/T) * RCH3**1.0 * PLIGM2**1.0  -1*A163 * T**n163 * np.exp(-1*E163/R/T) * PRKETM2**1.0 * PLIGM2**1.0  -1*A164 * T**n164 * np.exp(-1*E164/R/T) * RKET**1.0 * PLIGM2**1.0  -1*A165 * T**n165 * np.exp(-1*E165/R/T) * PRADIO**1.0 * PLIGM2**1.0  -1*A166 * T**n166 * np.exp(-1*E166/R/T) * RC3H3O**1.0 * PLIGM2**1.0  -1*A167 * T**n167 * np.exp(-1*E167/R/T) * RLIGB**1.0 * PLIGM2**1.0  -1*A168 * T**n168 * np.exp(-1*E168/R/T) * RLIGA**1.0 * PLIGM2**1.0  -1*A169 * T**n169 * np.exp(-1*E169/R/T) * PRADIOM2**1.0 * PLIGM2**1.0  -1*A170 * T**n170 * np.exp(-1*E170/R/T) * RMGUAI**1.0 * PLIGM2**1.0  -1*A171 * T**n171 * np.exp(-1*E171/R/T) * OH**1.0 * PLIGM2**1.0  -1*A172 * T**n172 * np.exp(-1*E172/R/T) * RCH3O**1.0 * PLIGM2**1.0  -1*A173 * T**n173 * np.exp(-1*E173/R/T) * RPHENOL**1.0 * PLIGM2**1.0  -1*A174 * T**n174 * np.exp(-1*E174/R/T) * RADIO**1.0 * PLIGM2**1.0, 
			-1*A31 * T**n31 * np.exp(-1*E31/R/T) * PLIGO**1.0  -1*A106 * T**n106 * np.exp(-1*E106/R/T) * PLIGO**1.0, 
			+1*A5 * T**n5 * np.exp(-1*E5/R/T) * PLIG**1.0  +1*A7 * T**n7 * np.exp(-1*E7/R/T) * PADIO**1.0  -2*A109 * T**n109 * np.exp(-1*E109/R/T) * PRADIO**2.0  -1*A123 * T**n123 * np.exp(-1*E123/R/T) * PRADIO**1.0 * LIGH**1.0  -1*A144 * T**n144 * np.exp(-1*E144/R/T) * PRADIO**1.0 * PLIGH**1.0  -1*A165 * T**n165 * np.exp(-1*E165/R/T) * PRADIO**1.0 * PLIGM2**1.0  -1*A186 * T**n186 * np.exp(-1*E186/R/T) * PRADIO**1.0 * LIGM2**1.0  -1*A207 * T**n207 * np.exp(-1*E207/R/T) * PRADIO**1.0 * LIGM2**1.0  -1*A228 * T**n228 * np.exp(-1*E228/R/T) * PRADIO**1.0 * PFET3M2**1.0  -1*A249 * T**n249 * np.exp(-1*E249/R/T) * PRADIO**1.0 * ADIOM2**1.0  -1*A270 * T**n270 * np.exp(-1*E270/R/T) * PRADIO**1.0 * KETM2**1.0  -1*A291 * T**n291 * np.exp(-1*E291/R/T) * PRADIO**1.0 * C10H2**1.0  -1*A312 * T**n312 * np.exp(-1*E312/R/T) * PRADIO**1.0 * LIG**1.0  -1*A333 * T**n333 * np.exp(-1*E333/R/T) * PRADIO**1.0 * LIG**1.0  -1*A354 * T**n354 * np.exp(-1*E354/R/T) * PRADIO**1.0 * PFET3**1.0  -1*A375 * T**n375 * np.exp(-1*E375/R/T) * PRADIO**1.0 * ADIO**1.0  -1*A396 * T**n396 * np.exp(-1*E396/R/T) * PRADIO**1.0 * KET**1.0, 
			+1*A3 * T**n3 * np.exp(-1*E3/R/T) * PLIGM2**1.0  +1*A6 * T**n6 * np.exp(-1*E6/R/T) * PADIOM2**1.0  -1*A58 * T**n58 * np.exp(-1*E58/R/T) * RCH3O**1.0 * PRADIOM2**1.0  -1*A71 * T**n71 * np.exp(-1*E71/R/T) * RCH3**1.0 * PRADIOM2**1.0  -2*A110 * T**n110 * np.exp(-1*E110/R/T) * PRADIOM2**2.0  -1*A127 * T**n127 * np.exp(-1*E127/R/T) * PRADIOM2**1.0 * LIGH**1.0  -1*A148 * T**n148 * np.exp(-1*E148/R/T) * PRADIOM2**1.0 * PLIGH**1.0  -1*A169 * T**n169 * np.exp(-1*E169/R/T) * PRADIOM2**1.0 * PLIGM2**1.0  -1*A190 * T**n190 * np.exp(-1*E190/R/T) * PRADIOM2**1.0 * LIGM2**1.0  -1*A211 * T**n211 * np.exp(-1*E211/R/T) * PRADIOM2**1.0 * LIGM2**1.0  -1*A232 * T**n232 * np.exp(-1*E232/R/T) * PRADIOM2**1.0 * PFET3M2**1.0  -1*A253 * T**n253 * np.exp(-1*E253/R/T) * PRADIOM2**1.0 * ADIOM2**1.0  -1*A274 * T**n274 * np.exp(-1*E274/R/T) * PRADIOM2**1.0 * KETM2**1.0  -1*A295 * T**n295 * np.exp(-1*E295/R/T) * PRADIOM2**1.0 * C10H2**1.0  -1*A316 * T**n316 * np.exp(-1*E316/R/T) * PRADIOM2**1.0 * LIG**1.0  -1*A337 * T**n337 * np.exp(-1*E337/R/T) * PRADIOM2**1.0 * LIG**1.0  -1*A358 * T**n358 * np.exp(-1*E358/R/T) * PRADIOM2**1.0 * PFET3**1.0  -1*A379 * T**n379 * np.exp(-1*E379/R/T) * PRADIOM2**1.0 * ADIO**1.0  -1*A400 * T**n400 * np.exp(-1*E400/R/T) * PRADIOM2**1.0 * KET**1.0, 
			-1*A22 * T**n22 * np.exp(-1*E22/R/T) * PRFET3**1.0  -1*A78 * T**n78 * np.exp(-1*E78/R/T) * PRFET3**1.0 * PRFET3**1.0  -1*A78 * T**n78 * np.exp(-1*E78/R/T) * PRFET3**1.0 * PRFET3**1.0  -1*A113 * T**n113 * np.exp(-1*E113/R/T) * PRFET3**1.0 * LIGH**1.0  -1*A134 * T**n134 * np.exp(-1*E134/R/T) * PRFET3**1.0 * PLIGH**1.0  -1*A155 * T**n155 * np.exp(-1*E155/R/T) * PRFET3**1.0 * PLIGM2**1.0  -1*A176 * T**n176 * np.exp(-1*E176/R/T) * PRFET3**1.0 * LIGM2**1.0  -1*A197 * T**n197 * np.exp(-1*E197/R/T) * PRFET3**1.0 * LIGM2**1.0  -1*A218 * T**n218 * np.exp(-1*E218/R/T) * PRFET3**1.0 * PFET3M2**1.0  -1*A239 * T**n239 * np.exp(-1*E239/R/T) * PRFET3**1.0 * ADIOM2**1.0  -1*A260 * T**n260 * np.exp(-1*E260/R/T) * PRFET3**1.0 * KETM2**1.0  -1*A281 * T**n281 * np.exp(-1*E281/R/T) * PRFET3**1.0 * C10H2**1.0  -1*A302 * T**n302 * np.exp(-1*E302/R/T) * PRFET3**1.0 * LIG**1.0  -1*A323 * T**n323 * np.exp(-1*E323/R/T) * PRFET3**1.0 * LIG**1.0  +1*A343 * T**n343 * np.exp(-1*E343/R/T) * RC3H5O2**1.0 * PFET3**1.0  -1*A344 * T**n344 * np.exp(-1*E344/R/T) * PRFET3**1.0 * PFET3**1.0  +1*A344 * T**n344 * np.exp(-1*E344/R/T) * PRFET3**1.0 * PFET3**1.0  +1*A345 * T**n345 * np.exp(-1*E345/R/T) * RC3H7O2**1.0 * PFET3**1.0  +1*A346 * T**n346 * np.exp(-1*E346/R/T) * RADIOM2**1.0 * PFET3**1.0  +1*A347 * T**n347 * np.exp(-1*E347/R/T) * PRFET3M2**1.0 * PFET3**1.0  +1*A348 * T**n348 * np.exp(-1*E348/R/T) * PRLIGH**1.0 * PFET3**1.0  +1*A349 * T**n349 * np.exp(-1*E349/R/T) * RLIGM2B**1.0 * PFET3**1.0  +1*A350 * T**n350 * np.exp(-1*E350/R/T) * RLIGM2A**1.0 * PFET3**1.0  +1*A351 * T**n351 * np.exp(-1*E351/R/T) * RCH3**1.0 * PFET3**1.0  +1*A352 * T**n352 * np.exp(-1*E352/R/T) * PRKETM2**1.0 * PFET3**1.0  +1*A353 * T**n353 * np.exp(-1*E353/R/T) * RKET**1.0 * PFET3**1.0  +1*A354 * T**n354 * np.exp(-1*E354/R/T) * PRADIO**1.0 * PFET3**1.0  +1*A355 * T**n355 * np.exp(-1*E355/R/T) * RC3H3O**1.0 * PFET3**1.0  +1*A356 * T**n356 * np.exp(-1*E356/R/T) * RLIGB**1.0 * PFET3**1.0  +1*A357 * T**n357 * np.exp(-1*E357/R/T) * RLIGA**1.0 * PFET3**1.0  +1*A358 * T**n358 * np.exp(-1*E358/R/T) * PRADIOM2**1.0 * PFET3**1.0  +1*A359 * T**n359 * np.exp(-1*E359/R/T) * RMGUAI**1.0 * PFET3**1.0  +1*A360 * T**n360 * np.exp(-1*E360/R/T) * OH**1.0 * PFET3**1.0  +1*A361 * T**n361 * np.exp(-1*E361/R/T) * RCH3O**1.0 * PFET3**1.0  +1*A362 * T**n362 * np.exp(-1*E362/R/T) * RPHENOL**1.0 * PFET3**1.0  +1*A363 * T**n363 * np.exp(-1*E363/R/T) * RADIO**1.0 * PFET3**1.0  -1*A365 * T**n365 * np.exp(-1*E365/R/T) * PRFET3**1.0 * ADIO**1.0  -1*A386 * T**n386 * np.exp(-1*E386/R/T) * PRFET3**1.0 * KET**1.0, 
			-1*A18 * T**n18 * np.exp(-1*E18/R/T) * PRFET3M2**1.0  -1*A65 * T**n65 * np.exp(-1*E65/R/T) * PRFET3M2**1.0 * PRFET3M2**1.0  -1*A65 * T**n65 * np.exp(-1*E65/R/T) * PRFET3M2**1.0 * PRFET3M2**1.0  -1*A116 * T**n116 * np.exp(-1*E116/R/T) * PRFET3M2**1.0 * LIGH**1.0  -1*A137 * T**n137 * np.exp(-1*E137/R/T) * PRFET3M2**1.0 * PLIGH**1.0  -1*A158 * T**n158 * np.exp(-1*E158/R/T) * PRFET3M2**1.0 * PLIGM2**1.0  -1*A179 * T**n179 * np.exp(-1*E179/R/T) * PRFET3M2**1.0 * LIGM2**1.0  -1*A200 * T**n200 * np.exp(-1*E200/R/T) * PRFET3M2**1.0 * LIGM2**1.0  +1*A217 * T**n217 * np.exp(-1*E217/R/T) * RC3H5O2**1.0 * PFET3M2**1.0  +1*A218 * T**n218 * np.exp(-1*E218/R/T) * PRFET3**1.0 * PFET3M2**1.0  +1*A219 * T**n219 * np.exp(-1*E219/R/T) * RC3H7O2**1.0 * PFET3M2**1.0  +1*A220 * T**n220 * np.exp(-1*E220/R/T) * RADIOM2**1.0 * PFET3M2**1.0  -1*A221 * T**n221 * np.exp(-1*E221/R/T) * PRFET3M2**1.0 * PFET3M2**1.0  +1*A221 * T**n221 * np.exp(-1*E221/R/T) * PRFET3M2**1.0 * PFET3M2**1.0  +1*A222 * T**n222 * np.exp(-1*E222/R/T) * PRLIGH**1.0 * PFET3M2**1.0  +1*A223 * T**n223 * np.exp(-1*E223/R/T) * RLIGM2B**1.0 * PFET3M2**1.0  +1*A224 * T**n224 * np.exp(-1*E224/R/T) * RLIGM2A**1.0 * PFET3M2**1.0  +1*A225 * T**n225 * np.exp(-1*E225/R/T) * RCH3**1.0 * PFET3M2**1.0  +1*A226 * T**n226 * np.exp(-1*E226/R/T) * PRKETM2**1.0 * PFET3M2**1.0  +1*A227 * T**n227 * np.exp(-1*E227/R/T) * RKET**1.0 * PFET3M2**1.0  +1*A228 * T**n228 * np.exp(-1*E228/R/T) * PRADIO**1.0 * PFET3M2**1.0  +1*A229 * T**n229 * np.exp(-1*E229/R/T) * RC3H3O**1.0 * PFET3M2**1.0  +1*A230 * T**n230 * np.exp(-1*E230/R/T) * RLIGB**1.0 * PFET3M2**1.0  +1*A231 * T**n231 * np.exp(-1*E231/R/T) * RLIGA**1.0 * PFET3M2**1.0  +1*A232 * T**n232 * np.exp(-1*E232/R/T) * PRADIOM2**1.0 * PFET3M2**1.0  +1*A233 * T**n233 * np.exp(-1*E233/R/T) * RMGUAI**1.0 * PFET3M2**1.0  +1*A234 * T**n234 * np.exp(-1*E234/R/T) * OH**1.0 * PFET3M2**1.0  +1*A235 * T**n235 * np.exp(-1*E235/R/T) * RCH3O**1.0 * PFET3M2**1.0  +1*A236 * T**n236 * np.exp(-1*E236/R/T) * RPHENOL**1.0 * PFET3M2**1.0  +1*A237 * T**n237 * np.exp(-1*E237/R/T) * RADIO**1.0 * PFET3M2**1.0  -1*A242 * T**n242 * np.exp(-1*E242/R/T) * PRFET3M2**1.0 * ADIOM2**1.0  -1*A263 * T**n263 * np.exp(-1*E263/R/T) * PRFET3M2**1.0 * KETM2**1.0  -1*A284 * T**n284 * np.exp(-1*E284/R/T) * PRFET3M2**1.0 * C10H2**1.0  -1*A305 * T**n305 * np.exp(-1*E305/R/T) * PRFET3M2**1.0 * LIG**1.0  -1*A326 * T**n326 * np.exp(-1*E326/R/T) * PRFET3M2**1.0 * LIG**1.0  -1*A347 * T**n347 * np.exp(-1*E347/R/T) * PRFET3M2**1.0 * PFET3**1.0  -1*A368 * T**n368 * np.exp(-1*E368/R/T) * PRFET3M2**1.0 * ADIO**1.0  -1*A389 * T**n389 * np.exp(-1*E389/R/T) * PRFET3M2**1.0 * KET**1.0, 
			+1*A8 * T**n8 * np.exp(-1*E8/R/T) * PKETM2**1.0  -1*A9 * T**n9 * np.exp(-1*E9/R/T) * PRKETM2**1.0  -1*A59 * T**n59 * np.exp(-1*E59/R/T) * RCH3O**1.0 * PRKETM2**1.0  -1*A72 * T**n72 * np.exp(-1*E72/R/T) * RCH3**1.0 * PRKETM2**1.0  -1*A121 * T**n121 * np.exp(-1*E121/R/T) * PRKETM2**1.0 * LIGH**1.0  -1*A142 * T**n142 * np.exp(-1*E142/R/T) * PRKETM2**1.0 * PLIGH**1.0  -1*A163 * T**n163 * np.exp(-1*E163/R/T) * PRKETM2**1.0 * PLIGM2**1.0  -1*A184 * T**n184 * np.exp(-1*E184/R/T) * PRKETM2**1.0 * LIGM2**1.0  -1*A205 * T**n205 * np.exp(-1*E205/R/T) * PRKETM2**1.0 * LIGM2**1.0  -1*A226 * T**n226 * np.exp(-1*E226/R/T) * PRKETM2**1.0 * PFET3M2**1.0  -1*A247 * T**n247 * np.exp(-1*E247/R/T) * PRKETM2**1.0 * ADIOM2**1.0  -1*A268 * T**n268 * np.exp(-1*E268/R/T) * PRKETM2**1.0 * KETM2**1.0  -1*A289 * T**n289 * np.exp(-1*E289/R/T) * PRKETM2**1.0 * C10H2**1.0  -1*A310 * T**n310 * np.exp(-1*E310/R/T) * PRKETM2**1.0 * LIG**1.0  -1*A331 * T**n331 * np.exp(-1*E331/R/T) * PRKETM2**1.0 * LIG**1.0  -1*A352 * T**n352 * np.exp(-1*E352/R/T) * PRKETM2**1.0 * PFET3**1.0  -1*A373 * T**n373 * np.exp(-1*E373/R/T) * PRKETM2**1.0 * ADIO**1.0  -1*A394 * T**n394 * np.exp(-1*E394/R/T) * PRKETM2**1.0 * KET**1.0, 
			+1*A0 * T**n0 * np.exp(-1*E0/R/T) * PLIGH**1.0  -1*A117 * T**n117 * np.exp(-1*E117/R/T) * PRLIGH**1.0 * LIGH**1.0  -1*A138 * T**n138 * np.exp(-1*E138/R/T) * PRLIGH**1.0 * PLIGH**1.0  -1*A159 * T**n159 * np.exp(-1*E159/R/T) * PRLIGH**1.0 * PLIGM2**1.0  -1*A180 * T**n180 * np.exp(-1*E180/R/T) * PRLIGH**1.0 * LIGM2**1.0  -1*A201 * T**n201 * np.exp(-1*E201/R/T) * PRLIGH**1.0 * LIGM2**1.0  -1*A222 * T**n222 * np.exp(-1*E222/R/T) * PRLIGH**1.0 * PFET3M2**1.0  -1*A243 * T**n243 * np.exp(-1*E243/R/T) * PRLIGH**1.0 * ADIOM2**1.0  -1*A264 * T**n264 * np.exp(-1*E264/R/T) * PRLIGH**1.0 * KETM2**1.0  -1*A285 * T**n285 * np.exp(-1*E285/R/T) * PRLIGH**1.0 * C10H2**1.0  -1*A306 * T**n306 * np.exp(-1*E306/R/T) * PRLIGH**1.0 * LIG**1.0  -1*A327 * T**n327 * np.exp(-1*E327/R/T) * PRLIGH**1.0 * LIG**1.0  -1*A348 * T**n348 * np.exp(-1*E348/R/T) * PRLIGH**1.0 * PFET3**1.0  -1*A369 * T**n369 * np.exp(-1*E369/R/T) * PRLIGH**1.0 * ADIO**1.0  -1*A390 * T**n390 * np.exp(-1*E390/R/T) * PRLIGH**1.0 * KET**1.0, 
			-1*A13 * T**n13 * np.exp(-1*E13/R/T) * PRLIGH2**1.0  +1*A133 * T**n133 * np.exp(-1*E133/R/T) * RC3H5O2**1.0 * PLIGH**1.0  +1*A134 * T**n134 * np.exp(-1*E134/R/T) * PRFET3**1.0 * PLIGH**1.0  +1*A135 * T**n135 * np.exp(-1*E135/R/T) * RC3H7O2**1.0 * PLIGH**1.0  +1*A136 * T**n136 * np.exp(-1*E136/R/T) * RADIOM2**1.0 * PLIGH**1.0  +1*A137 * T**n137 * np.exp(-1*E137/R/T) * PRFET3M2**1.0 * PLIGH**1.0  +1*A138 * T**n138 * np.exp(-1*E138/R/T) * PRLIGH**1.0 * PLIGH**1.0  +1*A139 * T**n139 * np.exp(-1*E139/R/T) * RLIGM2B**1.0 * PLIGH**1.0  +1*A140 * T**n140 * np.exp(-1*E140/R/T) * RLIGM2A**1.0 * PLIGH**1.0  +1*A141 * T**n141 * np.exp(-1*E141/R/T) * RCH3**1.0 * PLIGH**1.0  +1*A142 * T**n142 * np.exp(-1*E142/R/T) * PRKETM2**1.0 * PLIGH**1.0  +1*A143 * T**n143 * np.exp(-1*E143/R/T) * RKET**1.0 * PLIGH**1.0  +1*A144 * T**n144 * np.exp(-1*E144/R/T) * PRADIO**1.0 * PLIGH**1.0  +1*A145 * T**n145 * np.exp(-1*E145/R/T) * RC3H3O**1.0 * PLIGH**1.0  +1*A146 * T**n146 * np.exp(-1*E146/R/T) * RLIGB**1.0 * PLIGH**1.0  +1*A147 * T**n147 * np.exp(-1*E147/R/T) * RLIGA**1.0 * PLIGH**1.0  +1*A148 * T**n148 * np.exp(-1*E148/R/T) * PRADIOM2**1.0 * PLIGH**1.0  +1*A149 * T**n149 * np.exp(-1*E149/R/T) * RMGUAI**1.0 * PLIGH**1.0  +1*A150 * T**n150 * np.exp(-1*E150/R/T) * OH**1.0 * PLIGH**1.0  +1*A151 * T**n151 * np.exp(-1*E151/R/T) * RCH3O**1.0 * PLIGH**1.0  +1*A152 * T**n152 * np.exp(-1*E152/R/T) * RPHENOL**1.0 * PLIGH**1.0  +1*A153 * T**n153 * np.exp(-1*E153/R/T) * RADIO**1.0 * PLIGH**1.0, 
			+1*A13 * T**n13 * np.exp(-1*E13/R/T) * PRLIGH2**1.0  -1*A16 * T**n16 * np.exp(-1*E16/R/T) * PRLIGM2A**1.0  +1*A154 * T**n154 * np.exp(-1*E154/R/T) * RC3H5O2**1.0 * PLIGM2**1.0  +1*A155 * T**n155 * np.exp(-1*E155/R/T) * PRFET3**1.0 * PLIGM2**1.0  +1*A156 * T**n156 * np.exp(-1*E156/R/T) * RC3H7O2**1.0 * PLIGM2**1.0  +1*A157 * T**n157 * np.exp(-1*E157/R/T) * RADIOM2**1.0 * PLIGM2**1.0  +1*A158 * T**n158 * np.exp(-1*E158/R/T) * PRFET3M2**1.0 * PLIGM2**1.0  +1*A159 * T**n159 * np.exp(-1*E159/R/T) * PRLIGH**1.0 * PLIGM2**1.0  +1*A160 * T**n160 * np.exp(-1*E160/R/T) * RLIGM2B**1.0 * PLIGM2**1.0  +1*A161 * T**n161 * np.exp(-1*E161/R/T) * RLIGM2A**1.0 * PLIGM2**1.0  +1*A162 * T**n162 * np.exp(-1*E162/R/T) * RCH3**1.0 * PLIGM2**1.0  +1*A163 * T**n163 * np.exp(-1*E163/R/T) * PRKETM2**1.0 * PLIGM2**1.0  +1*A164 * T**n164 * np.exp(-1*E164/R/T) * RKET**1.0 * PLIGM2**1.0  +1*A165 * T**n165 * np.exp(-1*E165/R/T) * PRADIO**1.0 * PLIGM2**1.0  +1*A166 * T**n166 * np.exp(-1*E166/R/T) * RC3H3O**1.0 * PLIGM2**1.0  +1*A167 * T**n167 * np.exp(-1*E167/R/T) * RLIGB**1.0 * PLIGM2**1.0  +1*A168 * T**n168 * np.exp(-1*E168/R/T) * RLIGA**1.0 * PLIGM2**1.0  +1*A169 * T**n169 * np.exp(-1*E169/R/T) * PRADIOM2**1.0 * PLIGM2**1.0  +1*A170 * T**n170 * np.exp(-1*E170/R/T) * RMGUAI**1.0 * PLIGM2**1.0  +1*A171 * T**n171 * np.exp(-1*E171/R/T) * OH**1.0 * PLIGM2**1.0  +1*A172 * T**n172 * np.exp(-1*E172/R/T) * RCH3O**1.0 * PLIGM2**1.0  +1*A173 * T**n173 * np.exp(-1*E173/R/T) * RPHENOL**1.0 * PLIGM2**1.0  +1*A174 * T**n174 * np.exp(-1*E174/R/T) * RADIO**1.0 * PLIGM2**1.0, 
			+1*A4 * T**n4 * np.exp(-1*E4/R/T) * LIG**1.0  -1*A19 * T**n19 * np.exp(-1*E19/R/T) * RADIO**1.0  -1*A25 * T**n25 * np.exp(-1*E25/R/T) * RADIO**1.0  -2*A74 * T**n74 * np.exp(-1*E74/R/T) * RADIO**2.0  -1*A132 * T**n132 * np.exp(-1*E132/R/T) * RADIO**1.0 * LIGH**1.0  -1*A153 * T**n153 * np.exp(-1*E153/R/T) * RADIO**1.0 * PLIGH**1.0  -1*A174 * T**n174 * np.exp(-1*E174/R/T) * RADIO**1.0 * PLIGM2**1.0  -1*A195 * T**n195 * np.exp(-1*E195/R/T) * RADIO**1.0 * LIGM2**1.0  -1*A216 * T**n216 * np.exp(-1*E216/R/T) * RADIO**1.0 * LIGM2**1.0  -1*A237 * T**n237 * np.exp(-1*E237/R/T) * RADIO**1.0 * PFET3M2**1.0  -1*A258 * T**n258 * np.exp(-1*E258/R/T) * RADIO**1.0 * ADIOM2**1.0  -1*A279 * T**n279 * np.exp(-1*E279/R/T) * RADIO**1.0 * KETM2**1.0  -1*A300 * T**n300 * np.exp(-1*E300/R/T) * RADIO**1.0 * C10H2**1.0  -1*A321 * T**n321 * np.exp(-1*E321/R/T) * RADIO**1.0 * LIG**1.0  -1*A342 * T**n342 * np.exp(-1*E342/R/T) * RADIO**1.0 * LIG**1.0  -1*A363 * T**n363 * np.exp(-1*E363/R/T) * RADIO**1.0 * PFET3**1.0  +1*A364 * T**n364 * np.exp(-1*E364/R/T) * RC3H5O2**1.0 * ADIO**1.0  +1*A365 * T**n365 * np.exp(-1*E365/R/T) * PRFET3**1.0 * ADIO**1.0  +1*A366 * T**n366 * np.exp(-1*E366/R/T) * RC3H7O2**1.0 * ADIO**1.0  +1*A367 * T**n367 * np.exp(-1*E367/R/T) * RADIOM2**1.0 * ADIO**1.0  +1*A368 * T**n368 * np.exp(-1*E368/R/T) * PRFET3M2**1.0 * ADIO**1.0  +1*A369 * T**n369 * np.exp(-1*E369/R/T) * PRLIGH**1.0 * ADIO**1.0  +1*A370 * T**n370 * np.exp(-1*E370/R/T) * RLIGM2B**1.0 * ADIO**1.0  +1*A371 * T**n371 * np.exp(-1*E371/R/T) * RLIGM2A**1.0 * ADIO**1.0  +1*A372 * T**n372 * np.exp(-1*E372/R/T) * RCH3**1.0 * ADIO**1.0  +1*A373 * T**n373 * np.exp(-1*E373/R/T) * PRKETM2**1.0 * ADIO**1.0  +1*A374 * T**n374 * np.exp(-1*E374/R/T) * RKET**1.0 * ADIO**1.0  +1*A375 * T**n375 * np.exp(-1*E375/R/T) * PRADIO**1.0 * ADIO**1.0  +1*A376 * T**n376 * np.exp(-1*E376/R/T) * RC3H3O**1.0 * ADIO**1.0  +1*A377 * T**n377 * np.exp(-1*E377/R/T) * RLIGB**1.0 * ADIO**1.0  +1*A378 * T**n378 * np.exp(-1*E378/R/T) * RLIGA**1.0 * ADIO**1.0  +1*A379 * T**n379 * np.exp(-1*E379/R/T) * PRADIOM2**1.0 * ADIO**1.0  +1*A380 * T**n380 * np.exp(-1*E380/R/T) * RMGUAI**1.0 * ADIO**1.0  +1*A381 * T**n381 * np.exp(-1*E381/R/T) * OH**1.0 * ADIO**1.0  +1*A382 * T**n382 * np.exp(-1*E382/R/T) * RCH3O**1.0 * ADIO**1.0  +1*A383 * T**n383 * np.exp(-1*E383/R/T) * RPHENOL**1.0 * ADIO**1.0  -1*A384 * T**n384 * np.exp(-1*E384/R/T) * RADIO**1.0 * ADIO**1.0  +1*A384 * T**n384 * np.exp(-1*E384/R/T) * RADIO**1.0 * ADIO**1.0  -1*A405 * T**n405 * np.exp(-1*E405/R/T) * RADIO**1.0 * KET**1.0, 
			+1*A2 * T**n2 * np.exp(-1*E2/R/T) * LIGM2**1.0  -1*A14 * T**n14 * np.exp(-1*E14/R/T) * RADIOM2**1.0  -1*A23 * T**n23 * np.exp(-1*E23/R/T) * RADIOM2**1.0  -2*A60 * T**n60 * np.exp(-1*E60/R/T) * RADIOM2**2.0  -1*A115 * T**n115 * np.exp(-1*E115/R/T) * RADIOM2**1.0 * LIGH**1.0  -1*A136 * T**n136 * np.exp(-1*E136/R/T) * RADIOM2**1.0 * PLIGH**1.0  -1*A157 * T**n157 * np.exp(-1*E157/R/T) * RADIOM2**1.0 * PLIGM2**1.0  -1*A178 * T**n178 * np.exp(-1*E178/R/T) * RADIOM2**1.0 * LIGM2**1.0  -1*A199 * T**n199 * np.exp(-1*E199/R/T) * RADIOM2**1.0 * LIGM2**1.0  -1*A220 * T**n220 * np.exp(-1*E220/R/T) * RADIOM2**1.0 * PFET3M2**1.0  +1*A238 * T**n238 * np.exp(-1*E238/R/T) * RC3H5O2**1.0 * ADIOM2**1.0  +1*A239 * T**n239 * np.exp(-1*E239/R/T) * PRFET3**1.0 * ADIOM2**1.0  +1*A240 * T**n240 * np.exp(-1*E240/R/T) * RC3H7O2**1.0 * ADIOM2**1.0  -1*A241 * T**n241 * np.exp(-1*E241/R/T) * RADIOM2**1.0 * ADIOM2**1.0  +1*A241 * T**n241 * np.exp(-1*E241/R/T) * RADIOM2**1.0 * ADIOM2**1.0  +1*A242 * T**n242 * np.exp(-1*E242/R/T) * PRFET3M2**1.0 * ADIOM2**1.0  +1*A243 * T**n243 * np.exp(-1*E243/R/T) * PRLIGH**1.0 * ADIOM2**1.0  +1*A244 * T**n244 * np.exp(-1*E244/R/T) * RLIGM2B**1.0 * ADIOM2**1.0  +1*A245 * T**n245 * np.exp(-1*E245/R/T) * RLIGM2A**1.0 * ADIOM2**1.0  +1*A246 * T**n246 * np.exp(-1*E246/R/T) * RCH3**1.0 * ADIOM2**1.0  +1*A247 * T**n247 * np.exp(-1*E247/R/T) * PRKETM2**1.0 * ADIOM2**1.0  +1*A248 * T**n248 * np.exp(-1*E248/R/T) * RKET**1.0 * ADIOM2**1.0  +1*A249 * T**n249 * np.exp(-1*E249/R/T) * PRADIO**1.0 * ADIOM2**1.0  +1*A250 * T**n250 * np.exp(-1*E250/R/T) * RC3H3O**1.0 * ADIOM2**1.0  +1*A251 * T**n251 * np.exp(-1*E251/R/T) * RLIGB**1.0 * ADIOM2**1.0  +1*A252 * T**n252 * np.exp(-1*E252/R/T) * RLIGA**1.0 * ADIOM2**1.0  +1*A253 * T**n253 * np.exp(-1*E253/R/T) * PRADIOM2**1.0 * ADIOM2**1.0  +1*A254 * T**n254 * np.exp(-1*E254/R/T) * RMGUAI**1.0 * ADIOM2**1.0  +1*A255 * T**n255 * np.exp(-1*E255/R/T) * OH**1.0 * ADIOM2**1.0  +1*A256 * T**n256 * np.exp(-1*E256/R/T) * RCH3O**1.0 * ADIOM2**1.0  +1*A257 * T**n257 * np.exp(-1*E257/R/T) * RPHENOL**1.0 * ADIOM2**1.0  +1*A258 * T**n258 * np.exp(-1*E258/R/T) * RADIO**1.0 * ADIOM2**1.0  -1*A262 * T**n262 * np.exp(-1*E262/R/T) * RADIOM2**1.0 * KETM2**1.0  -1*A283 * T**n283 * np.exp(-1*E283/R/T) * RADIOM2**1.0 * C10H2**1.0  -1*A304 * T**n304 * np.exp(-1*E304/R/T) * RADIOM2**1.0 * LIG**1.0  -1*A325 * T**n325 * np.exp(-1*E325/R/T) * RADIOM2**1.0 * LIG**1.0  -1*A346 * T**n346 * np.exp(-1*E346/R/T) * RADIOM2**1.0 * PFET3**1.0  -1*A367 * T**n367 * np.exp(-1*E367/R/T) * RADIOM2**1.0 * ADIO**1.0  -1*A388 * T**n388 * np.exp(-1*E388/R/T) * RADIOM2**1.0 * KET**1.0, 
			+1*A34 * T**n34 * np.exp(-1*E34/R/T) * KETDM2**1.0 * RPHENOXM2**1.0  +1*A35 * T**n35 * np.exp(-1*E35/R/T) * SYNAPYL**1.0 * RPHENOXM2**1.0  +1*A38 * T**n38 * np.exp(-1*E38/R/T) * KETDM2**1.0 * RPHENOX**1.0  +1*A39 * T**n39 * np.exp(-1*E39/R/T) * SYNAPYL**1.0 * RPHENOX**1.0  +1*A42 * T**n42 * np.exp(-1*E42/R/T) * KETD**1.0 * RPHENOX**1.0  +1*A43 * T**n43 * np.exp(-1*E43/R/T) * COUMARYL**1.0 * RPHENOX**1.0  +1*A46 * T**n46 * np.exp(-1*E46/R/T) * KETD**1.0 * RPHENOXM2**1.0  +1*A47 * T**n47 * np.exp(-1*E47/R/T) * COUMARYL**1.0 * RPHENOXM2**1.0  -2*A68 * T**n68 * np.exp(-1*E68/R/T) * RC3H3O**2.0  -1*A81 * T**n81 * np.exp(-1*E81/R/T) * RPHENOX**1.0 * RC3H3O**1.0  -1*A124 * T**n124 * np.exp(-1*E124/R/T) * RC3H3O**1.0 * LIGH**1.0  -1*A145 * T**n145 * np.exp(-1*E145/R/T) * RC3H3O**1.0 * PLIGH**1.0  -1*A166 * T**n166 * np.exp(-1*E166/R/T) * RC3H3O**1.0 * PLIGM2**1.0  -1*A187 * T**n187 * np.exp(-1*E187/R/T) * RC3H3O**1.0 * LIGM2**1.0  -1*A208 * T**n208 * np.exp(-1*E208/R/T) * RC3H3O**1.0 * LIGM2**1.0  -1*A229 * T**n229 * np.exp(-1*E229/R/T) * RC3H3O**1.0 * PFET3M2**1.0  -1*A250 * T**n250 * np.exp(-1*E250/R/T) * RC3H3O**1.0 * ADIOM2**1.0  -1*A271 * T**n271 * np.exp(-1*E271/R/T) * RC3H3O**1.0 * KETM2**1.0  -1*A292 * T**n292 * np.exp(-1*E292/R/T) * RC3H3O**1.0 * C10H2**1.0  -1*A313 * T**n313 * np.exp(-1*E313/R/T) * RC3H3O**1.0 * LIG**1.0  -1*A334 * T**n334 * np.exp(-1*E334/R/T) * RC3H3O**1.0 * LIG**1.0  -1*A355 * T**n355 * np.exp(-1*E355/R/T) * RC3H3O**1.0 * PFET3**1.0  -1*A376 * T**n376 * np.exp(-1*E376/R/T) * RC3H3O**1.0 * ADIO**1.0  -1*A397 * T**n397 * np.exp(-1*E397/R/T) * RC3H3O**1.0 * KET**1.0, 
			+1*A33 * T**n33 * np.exp(-1*E33/R/T) * KETM2**1.0 * RPHENOXM2**1.0  +1*A37 * T**n37 * np.exp(-1*E37/R/T) * KETM2**1.0 * RPHENOX**1.0  +1*A41 * T**n41 * np.exp(-1*E41/R/T) * KET**1.0 * RPHENOX**1.0  +1*A45 * T**n45 * np.exp(-1*E45/R/T) * KET**1.0 * RPHENOXM2**1.0  -2*A67 * T**n67 * np.exp(-1*E67/R/T) * RC3H5O2**2.0  -1*A112 * T**n112 * np.exp(-1*E112/R/T) * RC3H5O2**1.0 * LIGH**1.0  -1*A133 * T**n133 * np.exp(-1*E133/R/T) * RC3H5O2**1.0 * PLIGH**1.0  -1*A154 * T**n154 * np.exp(-1*E154/R/T) * RC3H5O2**1.0 * PLIGM2**1.0  -1*A175 * T**n175 * np.exp(-1*E175/R/T) * RC3H5O2**1.0 * LIGM2**1.0  -1*A196 * T**n196 * np.exp(-1*E196/R/T) * RC3H5O2**1.0 * LIGM2**1.0  -1*A217 * T**n217 * np.exp(-1*E217/R/T) * RC3H5O2**1.0 * PFET3M2**1.0  -1*A238 * T**n238 * np.exp(-1*E238/R/T) * RC3H5O2**1.0 * ADIOM2**1.0  -1*A259 * T**n259 * np.exp(-1*E259/R/T) * RC3H5O2**1.0 * KETM2**1.0  -1*A280 * T**n280 * np.exp(-1*E280/R/T) * RC3H5O2**1.0 * C10H2**1.0  -1*A301 * T**n301 * np.exp(-1*E301/R/T) * RC3H5O2**1.0 * LIG**1.0  -1*A322 * T**n322 * np.exp(-1*E322/R/T) * RC3H5O2**1.0 * LIG**1.0  -1*A343 * T**n343 * np.exp(-1*E343/R/T) * RC3H5O2**1.0 * PFET3**1.0  -1*A364 * T**n364 * np.exp(-1*E364/R/T) * RC3H5O2**1.0 * ADIO**1.0  -1*A385 * T**n385 * np.exp(-1*E385/R/T) * RC3H5O2**1.0 * KET**1.0, 
			-1*A27 * T**n27 * np.exp(-1*E27/R/T) * RC3H7O2**1.0  +1*A32 * T**n32 * np.exp(-1*E32/R/T) * ADIOM2**1.0 * RPHENOXM2**1.0  +1*A36 * T**n36 * np.exp(-1*E36/R/T) * ADIOM2**1.0 * RPHENOX**1.0  +1*A40 * T**n40 * np.exp(-1*E40/R/T) * ADIO**1.0 * RPHENOX**1.0  +1*A44 * T**n44 * np.exp(-1*E44/R/T) * ADIO**1.0 * RPHENOXM2**1.0  -2*A66 * T**n66 * np.exp(-1*E66/R/T) * RC3H7O2**2.0  -1*A114 * T**n114 * np.exp(-1*E114/R/T) * RC3H7O2**1.0 * LIGH**1.0  -1*A135 * T**n135 * np.exp(-1*E135/R/T) * RC3H7O2**1.0 * PLIGH**1.0  -1*A156 * T**n156 * np.exp(-1*E156/R/T) * RC3H7O2**1.0 * PLIGM2**1.0  -1*A177 * T**n177 * np.exp(-1*E177/R/T) * RC3H7O2**1.0 * LIGM2**1.0  -1*A198 * T**n198 * np.exp(-1*E198/R/T) * RC3H7O2**1.0 * LIGM2**1.0  -1*A219 * T**n219 * np.exp(-1*E219/R/T) * RC3H7O2**1.0 * PFET3M2**1.0  -1*A240 * T**n240 * np.exp(-1*E240/R/T) * RC3H7O2**1.0 * ADIOM2**1.0  -1*A261 * T**n261 * np.exp(-1*E261/R/T) * RC3H7O2**1.0 * KETM2**1.0  -1*A282 * T**n282 * np.exp(-1*E282/R/T) * RC3H7O2**1.0 * C10H2**1.0  -1*A303 * T**n303 * np.exp(-1*E303/R/T) * RC3H7O2**1.0 * LIG**1.0  -1*A324 * T**n324 * np.exp(-1*E324/R/T) * RC3H7O2**1.0 * LIG**1.0  -1*A345 * T**n345 * np.exp(-1*E345/R/T) * RC3H7O2**1.0 * PFET3**1.0  -1*A366 * T**n366 * np.exp(-1*E366/R/T) * RC3H7O2**1.0 * ADIO**1.0  -1*A387 * T**n387 * np.exp(-1*E387/R/T) * RC3H7O2**1.0 * KET**1.0, 
			+1*A52 * T**n52 * np.exp(-1*E52/R/T) * RCH3O**1.0 * RPHENOX**1.0  +1*A53 * T**n53 * np.exp(-1*E53/R/T) * RCH3O**1.0 * RPHENOXM2**1.0  -1*A54 * T**n54 * np.exp(-1*E54/R/T) * RPHENOXM2**1.0 * RCH3**1.0  -1*A55 * T**n55 * np.exp(-1*E55/R/T) * RPHENOX**1.0 * RCH3**1.0  -1*A56 * T**n56 * np.exp(-1*E56/R/T) * RCH3O**1.0 * RCH3**1.0  -1*A69 * T**n69 * np.exp(-1*E69/R/T) * OH**1.0 * RCH3**1.0  -2*A70 * T**n70 * np.exp(-1*E70/R/T) * RCH3**2.0  -1*A71 * T**n71 * np.exp(-1*E71/R/T) * RCH3**1.0 * PRADIOM2**1.0  -1*A72 * T**n72 * np.exp(-1*E72/R/T) * RCH3**1.0 * PRKETM2**1.0  +1*A101 * T**n101 * np.exp(-1*E101/R/T) * PCH3**1.0  -1*A120 * T**n120 * np.exp(-1*E120/R/T) * RCH3**1.0 * LIGH**1.0  -1*A141 * T**n141 * np.exp(-1*E141/R/T) * RCH3**1.0 * PLIGH**1.0  -1*A162 * T**n162 * np.exp(-1*E162/R/T) * RCH3**1.0 * PLIGM2**1.0  -1*A183 * T**n183 * np.exp(-1*E183/R/T) * RCH3**1.0 * LIGM2**1.0  -1*A204 * T**n204 * np.exp(-1*E204/R/T) * RCH3**1.0 * LIGM2**1.0  -1*A225 * T**n225 * np.exp(-1*E225/R/T) * RCH3**1.0 * PFET3M2**1.0  -1*A246 * T**n246 * np.exp(-1*E246/R/T) * RCH3**1.0 * ADIOM2**1.0  -1*A267 * T**n267 * np.exp(-1*E267/R/T) * RCH3**1.0 * KETM2**1.0  -1*A288 * T**n288 * np.exp(-1*E288/R/T) * RCH3**1.0 * C10H2**1.0  -1*A309 * T**n309 * np.exp(-1*E309/R/T) * RCH3**1.0 * LIG**1.0  -1*A330 * T**n330 * np.exp(-1*E330/R/T) * RCH3**1.0 * LIG**1.0  -1*A351 * T**n351 * np.exp(-1*E351/R/T) * RCH3**1.0 * PFET3**1.0  -1*A372 * T**n372 * np.exp(-1*E372/R/T) * RCH3**1.0 * ADIO**1.0  -1*A393 * T**n393 * np.exp(-1*E393/R/T) * RCH3**1.0 * KET**1.0, 
			+1*A27 * T**n27 * np.exp(-1*E27/R/T) * RC3H7O2**1.0  +1*A28 * T**n28 * np.exp(-1*E28/R/T) * C10H2M4**1.0  +1*A29 * T**n29 * np.exp(-1*E29/R/T) * C10H2M2**1.0  +1*A48 * T**n48 * np.exp(-1*E48/R/T) * C10H2M4**1.0 * RPHENOXM2**1.0  +1*A49 * T**n49 * np.exp(-1*E49/R/T) * C10H2M2**1.0 * RPHENOXM2**1.0  +1*A50 * T**n50 * np.exp(-1*E50/R/T) * C10H2M4**1.0 * RPHENOX**1.0  +1*A51 * T**n51 * np.exp(-1*E51/R/T) * C10H2M2**1.0 * RPHENOX**1.0  -1*A52 * T**n52 * np.exp(-1*E52/R/T) * RCH3O**1.0 * RPHENOX**1.0  -1*A53 * T**n53 * np.exp(-1*E53/R/T) * RCH3O**1.0 * RPHENOXM2**1.0  -1*A56 * T**n56 * np.exp(-1*E56/R/T) * RCH3O**1.0 * RCH3**1.0  -2*A57 * T**n57 * np.exp(-1*E57/R/T) * RCH3O**2.0  -1*A58 * T**n58 * np.exp(-1*E58/R/T) * RCH3O**1.0 * PRADIOM2**1.0  -1*A59 * T**n59 * np.exp(-1*E59/R/T) * RCH3O**1.0 * PRKETM2**1.0  -1*A130 * T**n130 * np.exp(-1*E130/R/T) * RCH3O**1.0 * LIGH**1.0  -1*A151 * T**n151 * np.exp(-1*E151/R/T) * RCH3O**1.0 * PLIGH**1.0  -1*A172 * T**n172 * np.exp(-1*E172/R/T) * RCH3O**1.0 * PLIGM2**1.0  -1*A193 * T**n193 * np.exp(-1*E193/R/T) * RCH3O**1.0 * LIGM2**1.0  -1*A214 * T**n214 * np.exp(-1*E214/R/T) * RCH3O**1.0 * LIGM2**1.0  -1*A235 * T**n235 * np.exp(-1*E235/R/T) * RCH3O**1.0 * PFET3M2**1.0  -1*A256 * T**n256 * np.exp(-1*E256/R/T) * RCH3O**1.0 * ADIOM2**1.0  -1*A277 * T**n277 * np.exp(-1*E277/R/T) * RCH3O**1.0 * KETM2**1.0  -1*A298 * T**n298 * np.exp(-1*E298/R/T) * RCH3O**1.0 * C10H2**1.0  -1*A319 * T**n319 * np.exp(-1*E319/R/T) * RCH3O**1.0 * LIG**1.0  -1*A340 * T**n340 * np.exp(-1*E340/R/T) * RCH3O**1.0 * LIG**1.0  -1*A361 * T**n361 * np.exp(-1*E361/R/T) * RCH3O**1.0 * PFET3**1.0  -1*A382 * T**n382 * np.exp(-1*E382/R/T) * RCH3O**1.0 * ADIO**1.0  -1*A403 * T**n403 * np.exp(-1*E403/R/T) * RCH3O**1.0 * KET**1.0, 
			-1*A26 * T**n26 * np.exp(-1*E26/R/T) * RKET**1.0  -2*A77 * T**n77 * np.exp(-1*E77/R/T) * RKET**2.0  -1*A122 * T**n122 * np.exp(-1*E122/R/T) * RKET**1.0 * LIGH**1.0  -1*A143 * T**n143 * np.exp(-1*E143/R/T) * RKET**1.0 * PLIGH**1.0  -1*A164 * T**n164 * np.exp(-1*E164/R/T) * RKET**1.0 * PLIGM2**1.0  -1*A185 * T**n185 * np.exp(-1*E185/R/T) * RKET**1.0 * LIGM2**1.0  -1*A206 * T**n206 * np.exp(-1*E206/R/T) * RKET**1.0 * LIGM2**1.0  -1*A227 * T**n227 * np.exp(-1*E227/R/T) * RKET**1.0 * PFET3M2**1.0  -1*A248 * T**n248 * np.exp(-1*E248/R/T) * RKET**1.0 * ADIOM2**1.0  -1*A269 * T**n269 * np.exp(-1*E269/R/T) * RKET**1.0 * KETM2**1.0  -1*A290 * T**n290 * np.exp(-1*E290/R/T) * RKET**1.0 * C10H2**1.0  -1*A311 * T**n311 * np.exp(-1*E311/R/T) * RKET**1.0 * LIG**1.0  -1*A332 * T**n332 * np.exp(-1*E332/R/T) * RKET**1.0 * LIG**1.0  -1*A353 * T**n353 * np.exp(-1*E353/R/T) * RKET**1.0 * PFET3**1.0  -1*A374 * T**n374 * np.exp(-1*E374/R/T) * RKET**1.0 * ADIO**1.0  +1*A385 * T**n385 * np.exp(-1*E385/R/T) * RC3H5O2**1.0 * KET**1.0  +1*A386 * T**n386 * np.exp(-1*E386/R/T) * PRFET3**1.0 * KET**1.0  +1*A387 * T**n387 * np.exp(-1*E387/R/T) * RC3H7O2**1.0 * KET**1.0  +1*A388 * T**n388 * np.exp(-1*E388/R/T) * RADIOM2**1.0 * KET**1.0  +1*A389 * T**n389 * np.exp(-1*E389/R/T) * PRFET3M2**1.0 * KET**1.0  +1*A390 * T**n390 * np.exp(-1*E390/R/T) * PRLIGH**1.0 * KET**1.0  +1*A391 * T**n391 * np.exp(-1*E391/R/T) * RLIGM2B**1.0 * KET**1.0  +1*A392 * T**n392 * np.exp(-1*E392/R/T) * RLIGM2A**1.0 * KET**1.0  +1*A393 * T**n393 * np.exp(-1*E393/R/T) * RCH3**1.0 * KET**1.0  +1*A394 * T**n394 * np.exp(-1*E394/R/T) * PRKETM2**1.0 * KET**1.0  -1*A395 * T**n395 * np.exp(-1*E395/R/T) * RKET**1.0 * KET**1.0  +1*A395 * T**n395 * np.exp(-1*E395/R/T) * RKET**1.0 * KET**1.0  +1*A396 * T**n396 * np.exp(-1*E396/R/T) * PRADIO**1.0 * KET**1.0  +1*A397 * T**n397 * np.exp(-1*E397/R/T) * RC3H3O**1.0 * KET**1.0  +1*A398 * T**n398 * np.exp(-1*E398/R/T) * RLIGB**1.0 * KET**1.0  +1*A399 * T**n399 * np.exp(-1*E399/R/T) * RLIGA**1.0 * KET**1.0  +1*A400 * T**n400 * np.exp(-1*E400/R/T) * PRADIOM2**1.0 * KET**1.0  +1*A401 * T**n401 * np.exp(-1*E401/R/T) * RMGUAI**1.0 * KET**1.0  +1*A402 * T**n402 * np.exp(-1*E402/R/T) * OH**1.0 * KET**1.0  +1*A403 * T**n403 * np.exp(-1*E403/R/T) * RCH3O**1.0 * KET**1.0  +1*A404 * T**n404 * np.exp(-1*E404/R/T) * RPHENOL**1.0 * KET**1.0  +1*A405 * T**n405 * np.exp(-1*E405/R/T) * RADIO**1.0 * KET**1.0, 
			-1*A24 * T**n24 * np.exp(-1*E24/R/T) * RKETM2**1.0  -2*A64 * T**n64 * np.exp(-1*E64/R/T) * RKETM2**2.0  +1*A259 * T**n259 * np.exp(-1*E259/R/T) * RC3H5O2**1.0 * KETM2**1.0  +1*A260 * T**n260 * np.exp(-1*E260/R/T) * PRFET3**1.0 * KETM2**1.0  +1*A261 * T**n261 * np.exp(-1*E261/R/T) * RC3H7O2**1.0 * KETM2**1.0  +1*A262 * T**n262 * np.exp(-1*E262/R/T) * RADIOM2**1.0 * KETM2**1.0  +1*A263 * T**n263 * np.exp(-1*E263/R/T) * PRFET3M2**1.0 * KETM2**1.0  +1*A264 * T**n264 * np.exp(-1*E264/R/T) * PRLIGH**1.0 * KETM2**1.0  +1*A265 * T**n265 * np.exp(-1*E265/R/T) * RLIGM2B**1.0 * KETM2**1.0  +1*A266 * T**n266 * np.exp(-1*E266/R/T) * RLIGM2A**1.0 * KETM2**1.0  +1*A267 * T**n267 * np.exp(-1*E267/R/T) * RCH3**1.0 * KETM2**1.0  +1*A268 * T**n268 * np.exp(-1*E268/R/T) * PRKETM2**1.0 * KETM2**1.0  +1*A269 * T**n269 * np.exp(-1*E269/R/T) * RKET**1.0 * KETM2**1.0  +1*A270 * T**n270 * np.exp(-1*E270/R/T) * PRADIO**1.0 * KETM2**1.0  +1*A271 * T**n271 * np.exp(-1*E271/R/T) * RC3H3O**1.0 * KETM2**1.0  +1*A272 * T**n272 * np.exp(-1*E272/R/T) * RLIGB**1.0 * KETM2**1.0  +1*A273 * T**n273 * np.exp(-1*E273/R/T) * RLIGA**1.0 * KETM2**1.0  +1*A274 * T**n274 * np.exp(-1*E274/R/T) * PRADIOM2**1.0 * KETM2**1.0  +1*A275 * T**n275 * np.exp(-1*E275/R/T) * RMGUAI**1.0 * KETM2**1.0  +1*A276 * T**n276 * np.exp(-1*E276/R/T) * OH**1.0 * KETM2**1.0  +1*A277 * T**n277 * np.exp(-1*E277/R/T) * RCH3O**1.0 * KETM2**1.0  +1*A278 * T**n278 * np.exp(-1*E278/R/T) * RPHENOL**1.0 * KETM2**1.0  +1*A279 * T**n279 * np.exp(-1*E279/R/T) * RADIO**1.0 * KETM2**1.0, 
			-1*A20 * T**n20 * np.exp(-1*E20/R/T) * RLIGA**1.0  -2*A76 * T**n76 * np.exp(-1*E76/R/T) * RLIGA**2.0  -1*A126 * T**n126 * np.exp(-1*E126/R/T) * RLIGA**1.0 * LIGH**1.0  -1*A147 * T**n147 * np.exp(-1*E147/R/T) * RLIGA**1.0 * PLIGH**1.0  -1*A168 * T**n168 * np.exp(-1*E168/R/T) * RLIGA**1.0 * PLIGM2**1.0  -1*A189 * T**n189 * np.exp(-1*E189/R/T) * RLIGA**1.0 * LIGM2**1.0  -1*A210 * T**n210 * np.exp(-1*E210/R/T) * RLIGA**1.0 * LIGM2**1.0  -1*A231 * T**n231 * np.exp(-1*E231/R/T) * RLIGA**1.0 * PFET3M2**1.0  -1*A252 * T**n252 * np.exp(-1*E252/R/T) * RLIGA**1.0 * ADIOM2**1.0  -1*A273 * T**n273 * np.exp(-1*E273/R/T) * RLIGA**1.0 * KETM2**1.0  -1*A294 * T**n294 * np.exp(-1*E294/R/T) * RLIGA**1.0 * C10H2**1.0  +1*A301 * T**n301 * np.exp(-1*E301/R/T) * RC3H5O2**1.0 * LIG**1.0  +1*A302 * T**n302 * np.exp(-1*E302/R/T) * PRFET3**1.0 * LIG**1.0  +1*A303 * T**n303 * np.exp(-1*E303/R/T) * RC3H7O2**1.0 * LIG**1.0  +1*A304 * T**n304 * np.exp(-1*E304/R/T) * RADIOM2**1.0 * LIG**1.0  +1*A305 * T**n305 * np.exp(-1*E305/R/T) * PRFET3M2**1.0 * LIG**1.0  +1*A306 * T**n306 * np.exp(-1*E306/R/T) * PRLIGH**1.0 * LIG**1.0  +1*A307 * T**n307 * np.exp(-1*E307/R/T) * RLIGM2B**1.0 * LIG**1.0  +1*A308 * T**n308 * np.exp(-1*E308/R/T) * RLIGM2A**1.0 * LIG**1.0  +1*A309 * T**n309 * np.exp(-1*E309/R/T) * RCH3**1.0 * LIG**1.0  +1*A310 * T**n310 * np.exp(-1*E310/R/T) * PRKETM2**1.0 * LIG**1.0  +1*A311 * T**n311 * np.exp(-1*E311/R/T) * RKET**1.0 * LIG**1.0  +1*A312 * T**n312 * np.exp(-1*E312/R/T) * PRADIO**1.0 * LIG**1.0  +1*A313 * T**n313 * np.exp(-1*E313/R/T) * RC3H3O**1.0 * LIG**1.0  +1*A314 * T**n314 * np.exp(-1*E314/R/T) * RLIGB**1.0 * LIG**1.0  -1*A315 * T**n315 * np.exp(-1*E315/R/T) * RLIGA**1.0 * LIG**1.0  +1*A315 * T**n315 * np.exp(-1*E315/R/T) * RLIGA**1.0 * LIG**1.0  +1*A316 * T**n316 * np.exp(-1*E316/R/T) * PRADIOM2**1.0 * LIG**1.0  +1*A317 * T**n317 * np.exp(-1*E317/R/T) * RMGUAI**1.0 * LIG**1.0  +1*A318 * T**n318 * np.exp(-1*E318/R/T) * OH**1.0 * LIG**1.0  +1*A319 * T**n319 * np.exp(-1*E319/R/T) * RCH3O**1.0 * LIG**1.0  +1*A320 * T**n320 * np.exp(-1*E320/R/T) * RPHENOL**1.0 * LIG**1.0  +1*A321 * T**n321 * np.exp(-1*E321/R/T) * RADIO**1.0 * LIG**1.0  -1*A336 * T**n336 * np.exp(-1*E336/R/T) * RLIGA**1.0 * LIG**1.0  -1*A357 * T**n357 * np.exp(-1*E357/R/T) * RLIGA**1.0 * PFET3**1.0  -1*A378 * T**n378 * np.exp(-1*E378/R/T) * RLIGA**1.0 * ADIO**1.0  -1*A399 * T**n399 * np.exp(-1*E399/R/T) * RLIGA**1.0 * KET**1.0, 
			-1*A21 * T**n21 * np.exp(-1*E21/R/T) * RLIGB**1.0  -1*A73 * T**n73 * np.exp(-1*E73/R/T) * RPHENOX**1.0 * RLIGB**1.0  -1*A75 * T**n75 * np.exp(-1*E75/R/T) * RLIGB**1.0 * RLIGB**1.0  -1*A75 * T**n75 * np.exp(-1*E75/R/T) * RLIGB**1.0 * RLIGB**1.0  -1*A125 * T**n125 * np.exp(-1*E125/R/T) * RLIGB**1.0 * LIGH**1.0  -1*A146 * T**n146 * np.exp(-1*E146/R/T) * RLIGB**1.0 * PLIGH**1.0  -1*A167 * T**n167 * np.exp(-1*E167/R/T) * RLIGB**1.0 * PLIGM2**1.0  -1*A188 * T**n188 * np.exp(-1*E188/R/T) * RLIGB**1.0 * LIGM2**1.0  -1*A209 * T**n209 * np.exp(-1*E209/R/T) * RLIGB**1.0 * LIGM2**1.0  -1*A230 * T**n230 * np.exp(-1*E230/R/T) * RLIGB**1.0 * PFET3M2**1.0  -1*A251 * T**n251 * np.exp(-1*E251/R/T) * RLIGB**1.0 * ADIOM2**1.0  -1*A272 * T**n272 * np.exp(-1*E272/R/T) * RLIGB**1.0 * KETM2**1.0  -1*A293 * T**n293 * np.exp(-1*E293/R/T) * RLIGB**1.0 * C10H2**1.0  -1*A314 * T**n314 * np.exp(-1*E314/R/T) * RLIGB**1.0 * LIG**1.0  +1*A322 * T**n322 * np.exp(-1*E322/R/T) * RC3H5O2**1.0 * LIG**1.0  +1*A323 * T**n323 * np.exp(-1*E323/R/T) * PRFET3**1.0 * LIG**1.0  +1*A324 * T**n324 * np.exp(-1*E324/R/T) * RC3H7O2**1.0 * LIG**1.0  +1*A325 * T**n325 * np.exp(-1*E325/R/T) * RADIOM2**1.0 * LIG**1.0  +1*A326 * T**n326 * np.exp(-1*E326/R/T) * PRFET3M2**1.0 * LIG**1.0  +1*A327 * T**n327 * np.exp(-1*E327/R/T) * PRLIGH**1.0 * LIG**1.0  +1*A328 * T**n328 * np.exp(-1*E328/R/T) * RLIGM2B**1.0 * LIG**1.0  +1*A329 * T**n329 * np.exp(-1*E329/R/T) * RLIGM2A**1.0 * LIG**1.0  +1*A330 * T**n330 * np.exp(-1*E330/R/T) * RCH3**1.0 * LIG**1.0  +1*A331 * T**n331 * np.exp(-1*E331/R/T) * PRKETM2**1.0 * LIG**1.0  +1*A332 * T**n332 * np.exp(-1*E332/R/T) * RKET**1.0 * LIG**1.0  +1*A333 * T**n333 * np.exp(-1*E333/R/T) * PRADIO**1.0 * LIG**1.0  +1*A334 * T**n334 * np.exp(-1*E334/R/T) * RC3H3O**1.0 * LIG**1.0  -1*A335 * T**n335 * np.exp(-1*E335/R/T) * RLIGB**1.0 * LIG**1.0  +1*A335 * T**n335 * np.exp(-1*E335/R/T) * RLIGB**1.0 * LIG**1.0  +1*A336 * T**n336 * np.exp(-1*E336/R/T) * RLIGA**1.0 * LIG**1.0  +1*A337 * T**n337 * np.exp(-1*E337/R/T) * PRADIOM2**1.0 * LIG**1.0  +1*A338 * T**n338 * np.exp(-1*E338/R/T) * RMGUAI**1.0 * LIG**1.0  +1*A339 * T**n339 * np.exp(-1*E339/R/T) * OH**1.0 * LIG**1.0  +1*A340 * T**n340 * np.exp(-1*E340/R/T) * RCH3O**1.0 * LIG**1.0  +1*A341 * T**n341 * np.exp(-1*E341/R/T) * RPHENOL**1.0 * LIG**1.0  +1*A342 * T**n342 * np.exp(-1*E342/R/T) * RADIO**1.0 * LIG**1.0  -1*A356 * T**n356 * np.exp(-1*E356/R/T) * RLIGB**1.0 * PFET3**1.0  -1*A377 * T**n377 * np.exp(-1*E377/R/T) * RLIGB**1.0 * ADIO**1.0  -1*A398 * T**n398 * np.exp(-1*E398/R/T) * RLIGB**1.0 * KET**1.0, 
			-1*A12 * T**n12 * np.exp(-1*E12/R/T) * RLIGH**1.0  -1*A79 * T**n79 * np.exp(-1*E79/R/T) * RLIGH**1.0 * RLIGH**1.0  -1*A79 * T**n79 * np.exp(-1*E79/R/T) * RLIGH**1.0 * RLIGH**1.0  +1*A112 * T**n112 * np.exp(-1*E112/R/T) * RC3H5O2**1.0 * LIGH**1.0  +1*A113 * T**n113 * np.exp(-1*E113/R/T) * PRFET3**1.0 * LIGH**1.0  +1*A114 * T**n114 * np.exp(-1*E114/R/T) * RC3H7O2**1.0 * LIGH**1.0  +1*A115 * T**n115 * np.exp(-1*E115/R/T) * RADIOM2**1.0 * LIGH**1.0  +1*A116 * T**n116 * np.exp(-1*E116/R/T) * PRFET3M2**1.0 * LIGH**1.0  +1*A117 * T**n117 * np.exp(-1*E117/R/T) * PRLIGH**1.0 * LIGH**1.0  +1*A118 * T**n118 * np.exp(-1*E118/R/T) * RLIGM2B**1.0 * LIGH**1.0  +1*A119 * T**n119 * np.exp(-1*E119/R/T) * RLIGM2A**1.0 * LIGH**1.0  +1*A120 * T**n120 * np.exp(-1*E120/R/T) * RCH3**1.0 * LIGH**1.0  +1*A121 * T**n121 * np.exp(-1*E121/R/T) * PRKETM2**1.0 * LIGH**1.0  +1*A122 * T**n122 * np.exp(-1*E122/R/T) * RKET**1.0 * LIGH**1.0  +1*A123 * T**n123 * np.exp(-1*E123/R/T) * PRADIO**1.0 * LIGH**1.0  +1*A124 * T**n124 * np.exp(-1*E124/R/T) * RC3H3O**1.0 * LIGH**1.0  +1*A125 * T**n125 * np.exp(-1*E125/R/T) * RLIGB**1.0 * LIGH**1.0  +1*A126 * T**n126 * np.exp(-1*E126/R/T) * RLIGA**1.0 * LIGH**1.0  +1*A127 * T**n127 * np.exp(-1*E127/R/T) * PRADIOM2**1.0 * LIGH**1.0  +1*A128 * T**n128 * np.exp(-1*E128/R/T) * RMGUAI**1.0 * LIGH**1.0  +1*A129 * T**n129 * np.exp(-1*E129/R/T) * OH**1.0 * LIGH**1.0  +1*A130 * T**n130 * np.exp(-1*E130/R/T) * RCH3O**1.0 * LIGH**1.0  +1*A131 * T**n131 * np.exp(-1*E131/R/T) * RPHENOL**1.0 * LIGH**1.0  +1*A132 * T**n132 * np.exp(-1*E132/R/T) * RADIO**1.0 * LIGH**1.0, 
			+1*A1 * T**n1 * np.exp(-1*E1/R/T) * LIGH**1.0  +1*A12 * T**n12 * np.exp(-1*E12/R/T) * RLIGH**1.0  -1*A15 * T**n15 * np.exp(-1*E15/R/T) * RLIGM2A**1.0  -2*A62 * T**n62 * np.exp(-1*E62/R/T) * RLIGM2A**2.0  -1*A119 * T**n119 * np.exp(-1*E119/R/T) * RLIGM2A**1.0 * LIGH**1.0  -1*A140 * T**n140 * np.exp(-1*E140/R/T) * RLIGM2A**1.0 * PLIGH**1.0  -1*A161 * T**n161 * np.exp(-1*E161/R/T) * RLIGM2A**1.0 * PLIGM2**1.0  +1*A175 * T**n175 * np.exp(-1*E175/R/T) * RC3H5O2**1.0 * LIGM2**1.0  +1*A176 * T**n176 * np.exp(-1*E176/R/T) * PRFET3**1.0 * LIGM2**1.0  +1*A177 * T**n177 * np.exp(-1*E177/R/T) * RC3H7O2**1.0 * LIGM2**1.0  +1*A178 * T**n178 * np.exp(-1*E178/R/T) * RADIOM2**1.0 * LIGM2**1.0  +1*A179 * T**n179 * np.exp(-1*E179/R/T) * PRFET3M2**1.0 * LIGM2**1.0  +1*A180 * T**n180 * np.exp(-1*E180/R/T) * PRLIGH**1.0 * LIGM2**1.0  +1*A181 * T**n181 * np.exp(-1*E181/R/T) * RLIGM2B**1.0 * LIGM2**1.0  -1*A182 * T**n182 * np.exp(-1*E182/R/T) * RLIGM2A**1.0 * LIGM2**1.0  +1*A182 * T**n182 * np.exp(-1*E182/R/T) * RLIGM2A**1.0 * LIGM2**1.0  +1*A183 * T**n183 * np.exp(-1*E183/R/T) * RCH3**1.0 * LIGM2**1.0  +1*A184 * T**n184 * np.exp(-1*E184/R/T) * PRKETM2**1.0 * LIGM2**1.0  +1*A185 * T**n185 * np.exp(-1*E185/R/T) * RKET**1.0 * LIGM2**1.0  +1*A186 * T**n186 * np.exp(-1*E186/R/T) * PRADIO**1.0 * LIGM2**1.0  +1*A187 * T**n187 * np.exp(-1*E187/R/T) * RC3H3O**1.0 * LIGM2**1.0  +1*A188 * T**n188 * np.exp(-1*E188/R/T) * RLIGB**1.0 * LIGM2**1.0  +1*A189 * T**n189 * np.exp(-1*E189/R/T) * RLIGA**1.0 * LIGM2**1.0  +1*A190 * T**n190 * np.exp(-1*E190/R/T) * PRADIOM2**1.0 * LIGM2**1.0  +1*A191 * T**n191 * np.exp(-1*E191/R/T) * RMGUAI**1.0 * LIGM2**1.0  +1*A192 * T**n192 * np.exp(-1*E192/R/T) * OH**1.0 * LIGM2**1.0  +1*A193 * T**n193 * np.exp(-1*E193/R/T) * RCH3O**1.0 * LIGM2**1.0  +1*A194 * T**n194 * np.exp(-1*E194/R/T) * RPHENOL**1.0 * LIGM2**1.0  +1*A195 * T**n195 * np.exp(-1*E195/R/T) * RADIO**1.0 * LIGM2**1.0  -1*A203 * T**n203 * np.exp(-1*E203/R/T) * RLIGM2A**1.0 * LIGM2**1.0  -1*A224 * T**n224 * np.exp(-1*E224/R/T) * RLIGM2A**1.0 * PFET3M2**1.0  -1*A245 * T**n245 * np.exp(-1*E245/R/T) * RLIGM2A**1.0 * ADIOM2**1.0  -1*A266 * T**n266 * np.exp(-1*E266/R/T) * RLIGM2A**1.0 * KETM2**1.0  -1*A287 * T**n287 * np.exp(-1*E287/R/T) * RLIGM2A**1.0 * C10H2**1.0  -1*A308 * T**n308 * np.exp(-1*E308/R/T) * RLIGM2A**1.0 * LIG**1.0  -1*A329 * T**n329 * np.exp(-1*E329/R/T) * RLIGM2A**1.0 * LIG**1.0  -1*A350 * T**n350 * np.exp(-1*E350/R/T) * RLIGM2A**1.0 * PFET3**1.0  -1*A371 * T**n371 * np.exp(-1*E371/R/T) * RLIGM2A**1.0 * ADIO**1.0  -1*A392 * T**n392 * np.exp(-1*E392/R/T) * RLIGM2A**1.0 * KET**1.0, 
			-1*A17 * T**n17 * np.exp(-1*E17/R/T) * RLIGM2B**1.0  -2*A61 * T**n61 * np.exp(-1*E61/R/T) * RLIGM2B**2.0  -1*A118 * T**n118 * np.exp(-1*E118/R/T) * RLIGM2B**1.0 * LIGH**1.0  -1*A139 * T**n139 * np.exp(-1*E139/R/T) * RLIGM2B**1.0 * PLIGH**1.0  -1*A160 * T**n160 * np.exp(-1*E160/R/T) * RLIGM2B**1.0 * PLIGM2**1.0  -1*A181 * T**n181 * np.exp(-1*E181/R/T) * RLIGM2B**1.0 * LIGM2**1.0  +1*A196 * T**n196 * np.exp(-1*E196/R/T) * RC3H5O2**1.0 * LIGM2**1.0  +1*A197 * T**n197 * np.exp(-1*E197/R/T) * PRFET3**1.0 * LIGM2**1.0  +1*A198 * T**n198 * np.exp(-1*E198/R/T) * RC3H7O2**1.0 * LIGM2**1.0  +1*A199 * T**n199 * np.exp(-1*E199/R/T) * RADIOM2**1.0 * LIGM2**1.0  +1*A200 * T**n200 * np.exp(-1*E200/R/T) * PRFET3M2**1.0 * LIGM2**1.0  +1*A201 * T**n201 * np.exp(-1*E201/R/T) * PRLIGH**1.0 * LIGM2**1.0  -1*A202 * T**n202 * np.exp(-1*E202/R/T) * RLIGM2B**1.0 * LIGM2**1.0  +1*A202 * T**n202 * np.exp(-1*E202/R/T) * RLIGM2B**1.0 * LIGM2**1.0  +1*A203 * T**n203 * np.exp(-1*E203/R/T) * RLIGM2A**1.0 * LIGM2**1.0  +1*A204 * T**n204 * np.exp(-1*E204/R/T) * RCH3**1.0 * LIGM2**1.0  +1*A205 * T**n205 * np.exp(-1*E205/R/T) * PRKETM2**1.0 * LIGM2**1.0  +1*A206 * T**n206 * np.exp(-1*E206/R/T) * RKET**1.0 * LIGM2**1.0  +1*A207 * T**n207 * np.exp(-1*E207/R/T) * PRADIO**1.0 * LIGM2**1.0  +1*A208 * T**n208 * np.exp(-1*E208/R/T) * RC3H3O**1.0 * LIGM2**1.0  +1*A209 * T**n209 * np.exp(-1*E209/R/T) * RLIGB**1.0 * LIGM2**1.0  +1*A210 * T**n210 * np.exp(-1*E210/R/T) * RLIGA**1.0 * LIGM2**1.0  +1*A211 * T**n211 * np.exp(-1*E211/R/T) * PRADIOM2**1.0 * LIGM2**1.0  +1*A212 * T**n212 * np.exp(-1*E212/R/T) * RMGUAI**1.0 * LIGM2**1.0  +1*A213 * T**n213 * np.exp(-1*E213/R/T) * OH**1.0 * LIGM2**1.0  +1*A214 * T**n214 * np.exp(-1*E214/R/T) * RCH3O**1.0 * LIGM2**1.0  +1*A215 * T**n215 * np.exp(-1*E215/R/T) * RPHENOL**1.0 * LIGM2**1.0  +1*A216 * T**n216 * np.exp(-1*E216/R/T) * RADIO**1.0 * LIGM2**1.0  -1*A223 * T**n223 * np.exp(-1*E223/R/T) * RLIGM2B**1.0 * PFET3M2**1.0  -1*A244 * T**n244 * np.exp(-1*E244/R/T) * RLIGM2B**1.0 * ADIOM2**1.0  -1*A265 * T**n265 * np.exp(-1*E265/R/T) * RLIGM2B**1.0 * KETM2**1.0  -1*A286 * T**n286 * np.exp(-1*E286/R/T) * RLIGM2B**1.0 * C10H2**1.0  -1*A307 * T**n307 * np.exp(-1*E307/R/T) * RLIGM2B**1.0 * LIG**1.0  -1*A328 * T**n328 * np.exp(-1*E328/R/T) * RLIGM2B**1.0 * LIG**1.0  -1*A349 * T**n349 * np.exp(-1*E349/R/T) * RLIGM2B**1.0 * PFET3**1.0  -1*A370 * T**n370 * np.exp(-1*E370/R/T) * RLIGM2B**1.0 * ADIO**1.0  -1*A391 * T**n391 * np.exp(-1*E391/R/T) * RLIGM2B**1.0 * KET**1.0, 
			+1*A14 * T**n14 * np.exp(-1*E14/R/T) * RADIOM2**1.0  +1*A17 * T**n17 * np.exp(-1*E17/R/T) * RLIGM2B**1.0  -2*A63 * T**n63 * np.exp(-1*E63/R/T) * RMGUAI**2.0  -1*A128 * T**n128 * np.exp(-1*E128/R/T) * RMGUAI**1.0 * LIGH**1.0  -1*A149 * T**n149 * np.exp(-1*E149/R/T) * RMGUAI**1.0 * PLIGH**1.0  -1*A170 * T**n170 * np.exp(-1*E170/R/T) * RMGUAI**1.0 * PLIGM2**1.0  -1*A191 * T**n191 * np.exp(-1*E191/R/T) * RMGUAI**1.0 * LIGM2**1.0  -1*A212 * T**n212 * np.exp(-1*E212/R/T) * RMGUAI**1.0 * LIGM2**1.0  -1*A233 * T**n233 * np.exp(-1*E233/R/T) * RMGUAI**1.0 * PFET3M2**1.0  -1*A254 * T**n254 * np.exp(-1*E254/R/T) * RMGUAI**1.0 * ADIOM2**1.0  -1*A275 * T**n275 * np.exp(-1*E275/R/T) * RMGUAI**1.0 * KETM2**1.0  -1*A296 * T**n296 * np.exp(-1*E296/R/T) * RMGUAI**1.0 * C10H2**1.0  -1*A317 * T**n317 * np.exp(-1*E317/R/T) * RMGUAI**1.0 * LIG**1.0  -1*A338 * T**n338 * np.exp(-1*E338/R/T) * RMGUAI**1.0 * LIG**1.0  -1*A359 * T**n359 * np.exp(-1*E359/R/T) * RMGUAI**1.0 * PFET3**1.0  -1*A380 * T**n380 * np.exp(-1*E380/R/T) * RMGUAI**1.0 * ADIO**1.0  -1*A401 * T**n401 * np.exp(-1*E401/R/T) * RMGUAI**1.0 * KET**1.0, 
			+1*A19 * T**n19 * np.exp(-1*E19/R/T) * RADIO**1.0  +1*A21 * T**n21 * np.exp(-1*E21/R/T) * RLIGB**1.0  -1*A80 * T**n80 * np.exp(-1*E80/R/T) * RPHENOX**1.0 * RPHENOL**1.0  -1*A131 * T**n131 * np.exp(-1*E131/R/T) * RPHENOL**1.0 * LIGH**1.0  -1*A152 * T**n152 * np.exp(-1*E152/R/T) * RPHENOL**1.0 * PLIGH**1.0  -1*A173 * T**n173 * np.exp(-1*E173/R/T) * RPHENOL**1.0 * PLIGM2**1.0  -1*A194 * T**n194 * np.exp(-1*E194/R/T) * RPHENOL**1.0 * LIGM2**1.0  -1*A215 * T**n215 * np.exp(-1*E215/R/T) * RPHENOL**1.0 * LIGM2**1.0  -1*A236 * T**n236 * np.exp(-1*E236/R/T) * RPHENOL**1.0 * PFET3M2**1.0  -1*A257 * T**n257 * np.exp(-1*E257/R/T) * RPHENOL**1.0 * ADIOM2**1.0  -1*A278 * T**n278 * np.exp(-1*E278/R/T) * RPHENOL**1.0 * KETM2**1.0  -1*A299 * T**n299 * np.exp(-1*E299/R/T) * RPHENOL**1.0 * C10H2**1.0  -1*A320 * T**n320 * np.exp(-1*E320/R/T) * RPHENOL**1.0 * LIG**1.0  -1*A341 * T**n341 * np.exp(-1*E341/R/T) * RPHENOL**1.0 * LIG**1.0  -1*A362 * T**n362 * np.exp(-1*E362/R/T) * RPHENOL**1.0 * PFET3**1.0  -1*A383 * T**n383 * np.exp(-1*E383/R/T) * RPHENOL**1.0 * ADIO**1.0  -1*A404 * T**n404 * np.exp(-1*E404/R/T) * RPHENOL**1.0 * KET**1.0, 
			+1*A4 * T**n4 * np.exp(-1*E4/R/T) * LIG**1.0  +1*A5 * T**n5 * np.exp(-1*E5/R/T) * PLIG**1.0  -1*A11 * T**n11 * np.exp(-1*E11/R/T) * RPHENOX**1.0  +1*A20 * T**n20 * np.exp(-1*E20/R/T) * RLIGA**1.0  +1*A22 * T**n22 * np.exp(-1*E22/R/T) * PRFET3**1.0  +1*A28 * T**n28 * np.exp(-1*E28/R/T) * C10H2M4**1.0  +1*A29 * T**n29 * np.exp(-1*E29/R/T) * C10H2M2**1.0  -1*A36 * T**n36 * np.exp(-1*E36/R/T) * ADIOM2**1.0 * RPHENOX**1.0  -1*A37 * T**n37 * np.exp(-1*E37/R/T) * KETM2**1.0 * RPHENOX**1.0  -1*A38 * T**n38 * np.exp(-1*E38/R/T) * KETDM2**1.0 * RPHENOX**1.0  -1*A39 * T**n39 * np.exp(-1*E39/R/T) * SYNAPYL**1.0 * RPHENOX**1.0  -1*A40 * T**n40 * np.exp(-1*E40/R/T) * ADIO**1.0 * RPHENOX**1.0  -1*A41 * T**n41 * np.exp(-1*E41/R/T) * KET**1.0 * RPHENOX**1.0  -1*A42 * T**n42 * np.exp(-1*E42/R/T) * KETD**1.0 * RPHENOX**1.0  -1*A43 * T**n43 * np.exp(-1*E43/R/T) * COUMARYL**1.0 * RPHENOX**1.0  -1*A50 * T**n50 * np.exp(-1*E50/R/T) * C10H2M4**1.0 * RPHENOX**1.0  -1*A51 * T**n51 * np.exp(-1*E51/R/T) * C10H2M2**1.0 * RPHENOX**1.0  -1*A52 * T**n52 * np.exp(-1*E52/R/T) * RCH3O**1.0 * RPHENOX**1.0  -1*A55 * T**n55 * np.exp(-1*E55/R/T) * RPHENOX**1.0 * RCH3**1.0  -1*A73 * T**n73 * np.exp(-1*E73/R/T) * RPHENOX**1.0 * RLIGB**1.0  -1*A80 * T**n80 * np.exp(-1*E80/R/T) * RPHENOX**1.0 * RPHENOL**1.0  -1*A81 * T**n81 * np.exp(-1*E81/R/T) * RPHENOX**1.0 * RC3H3O**1.0  -1*A82 * T**n82 * np.exp(-1*E82/R/T) * RPHENOX**1.0 * CHAR**1.0, 
			+1*A2 * T**n2 * np.exp(-1*E2/R/T) * LIGM2**1.0  +1*A3 * T**n3 * np.exp(-1*E3/R/T) * PLIGM2**1.0  -1*A10 * T**n10 * np.exp(-1*E10/R/T) * RPHENOXM2**1.0  +1*A15 * T**n15 * np.exp(-1*E15/R/T) * RLIGM2A**1.0  +1*A16 * T**n16 * np.exp(-1*E16/R/T) * PRLIGM2A**1.0  +1*A18 * T**n18 * np.exp(-1*E18/R/T) * PRFET3M2**1.0  -1*A32 * T**n32 * np.exp(-1*E32/R/T) * ADIOM2**1.0 * RPHENOXM2**1.0  -1*A33 * T**n33 * np.exp(-1*E33/R/T) * KETM2**1.0 * RPHENOXM2**1.0  -1*A34 * T**n34 * np.exp(-1*E34/R/T) * KETDM2**1.0 * RPHENOXM2**1.0  -1*A35 * T**n35 * np.exp(-1*E35/R/T) * SYNAPYL**1.0 * RPHENOXM2**1.0  -1*A44 * T**n44 * np.exp(-1*E44/R/T) * ADIO**1.0 * RPHENOXM2**1.0  -1*A45 * T**n45 * np.exp(-1*E45/R/T) * KET**1.0 * RPHENOXM2**1.0  -1*A46 * T**n46 * np.exp(-1*E46/R/T) * KETD**1.0 * RPHENOXM2**1.0  -1*A47 * T**n47 * np.exp(-1*E47/R/T) * COUMARYL**1.0 * RPHENOXM2**1.0  -1*A48 * T**n48 * np.exp(-1*E48/R/T) * C10H2M4**1.0 * RPHENOXM2**1.0  -1*A49 * T**n49 * np.exp(-1*E49/R/T) * C10H2M2**1.0 * RPHENOXM2**1.0  -1*A53 * T**n53 * np.exp(-1*E53/R/T) * RCH3O**1.0 * RPHENOXM2**1.0  -1*A54 * T**n54 * np.exp(-1*E54/R/T) * RPHENOXM2**1.0 * RCH3**1.0  -1*A83 * T**n83 * np.exp(-1*E83/R/T) * RPHENOXM2**1.0 * CHAR**1.0, 
			+1*A23 * T**n23 * np.exp(-1*E23/R/T) * RADIOM2**1.0  -1*A35 * T**n35 * np.exp(-1*E35/R/T) * SYNAPYL**1.0 * RPHENOXM2**1.0  -1*A39 * T**n39 * np.exp(-1*E39/R/T) * SYNAPYL**1.0 * RPHENOX**1.0  -1*A87 * T**n87 * np.exp(-1*E87/R/T) * SYNAPYL**1.0, 
			+1*A90 * T**n90 * np.exp(-1*E90/R/T) * ADIO**1.0, 
			+1*A84 * T**n84 * np.exp(-1*E84/R/T) * ADIOM2**1.0, 
			+1*A89 * T**n89 * np.exp(-1*E89/R/T) * COUMARYL**1.0, 
			+1*A91 * T**n91 * np.exp(-1*E91/R/T) * KET**1.0, 
			+1*A92 * T**n92 * np.exp(-1*E92/R/T) * KETD**1.0, 
			+1*A86 * T**n86 * np.exp(-1*E86/R/T) * KETDM2**1.0, 
			+1*A85 * T**n85 * np.exp(-1*E85/R/T) * KETM2**1.0, 
			+1*A88 * T**n88 * np.exp(-1*E88/R/T) * MGUAI**1.0, 
			+1*A93 * T**n93 * np.exp(-1*E93/R/T) * PHENOL**1.0, 
			+1*A87 * T**n87 * np.exp(-1*E87/R/T) * SYNAPYL**1.0]
	return dydt

def run():
	start = time.time()
	R = 8.314
	# A, n, E values
	A0 = 1.00E+13
	n0 = 0
	E0 = 163254
	A1 = 1.00E+13
	n1 = 0
	E1 = 163254
	A2 = 1.00E+13
	n2 = 0
	E2 = 163254
	A3 = 1.00E+13
	n3 = 0
	E3 = 163254
	A4 = 1.00E+13
	n4 = 0
	E4 = 184184
	A5 = 1.00E+13
	n5 = 0
	E5 = 188370
	A6 = 1.00E+13
	n6 = 0
	E6 = 171626
	A7 = 1.00E+13
	n7 = 0
	E7 = 179998
	A8 = 1.00E+13
	n8 = 0
	E8 = 167440
	A9 = 1.00E+13
	n9 = 0
	E9 = 121394
	A10 = 4.00E+10
	n10 = 0
	E10 = 209300
	A11 = 4.00E+10
	n11 = 0
	E11 = 209300
	A12 = 1.00E+13
	n12 = 0
	E12 = 133952
	A13 = 1.00E+13
	n13 = 0
	E13 = 133952
	A14 = 1.00E+13
	n14 = 0
	E14 = 133952
	A15 = 5.00E+12
	n15 = 0
	E15 = 133952
	A16 = 5.00E+12
	n16 = 0
	E16 = 133952
	A17 = 1.00E+13
	n17 = 0
	E17 = 163254
	A18 = 1.00E+13
	n18 = 0
	E18 = 133952
	A19 = 1.00E+13
	n19 = 0
	E19 = 163254
	A20 = 1.00E+13
	n20 = 0
	E20 = 138138
	A21 = 1.00E+13
	n21 = 0
	E21 = 163254
	A22 = 1.00E+13
	n22 = 0
	E22 = 138138
	A23 = 3.00E+11
	n23 = 0
	E23 = 104650
	A24 = 3.00E+11
	n24 = 0
	E24 = 104650
	A25 = 3.00E+11
	n25 = 0
	E25 = 104650
	A26 = 3.00E+11
	n26 = 0
	E26 = 113022
	A27 = 1.00E+13
	n27 = 0
	E27 = 129766
	A28 = 1.00E+13
	n28 = 0
	E28 = 196742
	A29 = 1.00E+13
	n29 = 0
	E29 = 196742
	A30 = 1.00E+08
	n30 = 0
	E30 = 121394
	A31 = 1.00E+09
	n31 = 0
	E31 = 108836
	A32 = 1.00E+09
	n32 = 0
	E32 = 121394
	A33 = 1.00E+09
	n33 = 0
	E33 = 121394
	A34 = 1.00E+09
	n34 = 0
	E34 = 121394
	A35 = 1.00E+09
	n35 = 0
	E35 = 121394
	A36 = 1.00E+09
	n36 = 0
	E36 = 121394
	A37 = 1.00E+09
	n37 = 0
	E37 = 121394
	A38 = 1.00E+09
	n38 = 0
	E38 = 121394
	A39 = 1.00E+09
	n39 = 0
	E39 = 121394
	A40 = 1.00E+09
	n40 = 0
	E40 = 121394
	A41 = 1.00E+09
	n41 = 0
	E41 = 121394
	A42 = 1.00E+09
	n42 = 0
	E42 = 121394
	A43 = 1.00E+09
	n43 = 0
	E43 = 121394
	A44 = 1.00E+09
	n44 = 0
	E44 = 121394
	A45 = 1.00E+09
	n45 = 0
	E45 = 121394
	A46 = 1.00E+09
	n46 = 0
	E46 = 121394
	A47 = 1.00E+09
	n47 = 0
	E47 = 121394
	A48 = 1.00E+09
	n48 = 0
	E48 = 115115
	A49 = 1.00E+09
	n49 = 0
	E49 = 115115
	A50 = 1.00E+09
	n50 = 0
	E50 = 117208
	A51 = 1.00E+09
	n51 = 0
	E51 = 117208
	A52 = 1.00E+08
	n52 = 0
	E52 = 54418
	A53 = 1.00E+08
	n53 = 0
	E53 = 54418
	A54 = 1.00E+08
	n54 = 0
	E54 = 48139
	A55 = 1.00E+08
	n55 = 0
	E55 = 48139
	A56 = 1.00E+08
	n56 = 0
	E56 = 12558
	A57 = 1.00E+08
	n57 = 0
	E57 = 12558
	A58 = 1.00E+08
	n58 = 0
	E58 = 52325
	A59 = 1.00E+08
	n59 = 0
	E59 = 52325
	A60 = 3.16E+07
	n60 = 0
	E60 = 83720
	A61 = 3.16E+07
	n61 = 0
	E61 = 83720
	A62 = 3.16E+07
	n62 = 0
	E62 = 83720
	A63 = 3.16E+07
	n63 = 0
	E63 = 83720
	A64 = 3.16E+07
	n64 = 0
	E64 = 83720
	A65 = 3.16E+07
	n65 = 0
	E65 = 83720
	A66 = 1.00E+08
	n66 = 0
	E66 = 12558
	A67 = 1.00E+08
	n67 = 0
	E67 = 12558
	A68 = 1.00E+08
	n68 = 0
	E68 = 12558
	A69 = 1.00E+08
	n69 = 0
	E69 = 12558
	A70 = 1.00E+08
	n70 = 0
	E70 = 12558
	A71 = 1.00E+08
	n71 = 0
	E71 = 48139
	A72 = 1.00E+08
	n72 = 0
	E72 = 48139
	A73 = 3.16E+07
	n73 = 0
	E73 = 83720
	A74 = 3.16E+07
	n74 = 0
	E74 = 92092
	A75 = 3.16E+07
	n75 = 0
	E75 = 83720
	A76 = 3.16E+07
	n76 = 0
	E76 = 83720
	A77 = 3.16E+07
	n77 = 0
	E77 = 83720
	A78 = 3.16E+07
	n78 = 0
	E78 = 83720
	A79 = 3.16E+07
	n79 = 0
	E79 = 83720
	A80 = 5.00E+07
	n80 = 0
	E80 = 83720
	A81 = 1.00E+08
	n81 = 0
	E81 = 48139
	A82 = 3.00E+07
	n82 = 0
	E82 = 104650
	A83 = 3.00E+07
	n83 = 0
	E83 = 104650
	A84 = 1
	n84 = 1
	E84 = 62790
	A85 = 1
	n85 = 1
	E85 = 62790
	A86 = 1
	n86 = 1
	E86 = 58604
	A87 = 1
	n87 = 1
	E87 = 62790
	A88 = 1
	n88 = 1
	E88 = 66976
	A89 = 1
	n89 = 1
	E89 = 83720
	A90 = 1
	n90 = 1
	E90 = 66976
	A91 = 1
	n91 = 1
	E91 = 66976
	A92 = 1
	n92 = 1
	E92 = 83720
	A93 = 1
	n93 = 1
	E93 = 58604
	A94 = 2.00E+08
	n94 = 0
	E94 = 209300
	A95 = 1.00E+07
	n95 = 0
	E95 = 138138
	A96 = 1.00E+10
	n96 = 0
	E96 = 209300
	A97 = 5.00E+08
	n97 = 0
	E97 = 205114
	A98 = 5.00E+08
	n98 = 0
	E98 = 0
	A99 = 5.00E+08
	n99 = 0
	E99 = 0
	A100 = 5.00E+08
	n100 = 0
	E100 = 0
	A101 = 1.00E+13
	n101 = 0
	E101 = 125580
	A102 = 5.00E+08
	n102 = 0
	E102 = 209300
	A103 = 5.00E+08
	n103 = 0
	E103 = 209300
	A104 = 1E13
	n104 = 0
	E104 = 154812
	A105 = 1E13
	n105 = 0
	E105 = 168458
	A106 = 1E13
	n106 = 0
	E106 = 154812
	A107 = 1E9
	n107 = 0
	E107 = 108836
	A108 = 1E8
	n108 = 0
	E108 = 121394
	A109 = 3.16E7
	n109 = 0
	E109 = 92092
	A110 = 3.16E7
	n110 = 0
	E110 = 83720
	A111 = 1E7
	n111 = 0
	E111 = 138138
	A112 = 2*10**8
	n112 = 0
	E112 = 71162-4186
	A113 = 2*10**8
	n113 = 0
	E113 = 54418-4186
	A114 = 2*10**8
	n114 = 0
	E114 = 62790-4186
	A115 = 2*10**8
	n115 = 0
	E115 = 54418-4186
	A116 = 2*10**8
	n116 = 0
	E116 = 54418-4186
	A117 = 2*10**8.5
	n117 = 0
	E117 = 41860-4186
	A118 = 2*10**8
	n118 = 0
	E118 = 54418-4186
	A119 = 2*10**8
	n119 = 0
	E119 = 54418-4186
	A120 = 2*10**8.5
	n120 = 0
	E120 = 48139-4186
	A121 = 2*10**8
	n121 = 0
	E121 = 54418-4186
	A122 = 2*10**8.5
	n122 = 0
	E122 = 54418-4186
	A123 = 2*10**8
	n123 = 0
	E123 = 46046-4186
	A124 = 2*10**8
	n124 = 0
	E124 = 83720-4186
	A125 = 2*10**8
	n125 = 0
	E125 = 56511-4186
	A126 = 2*10**8
	n126 = 0
	E126 = 56511-4186
	A127 = 2*10**8.5
	n127 = 0
	E127 = 41860-4186
	A128 = 2*10**8
	n128 = 0
	E128 = 54418-4186
	A129 = 2*10**9.5
	n129 = 0
	E129 = 14651-4186
	A130 = 2*10**8.5
	n130 = 0
	E130 = 33069-4186
	A131 = 2*10**8.5
	n131 = 0
	E131 = 54418-4186
	A132 = 2*10**8
	n132 = 0
	E132 = 54418-4186
	A133 = 2*10**8
	n133 = 0
	E133 = 71162-4186
	A134 = 2*10**8
	n134 = 0
	E134 = 54418-4186
	A135 = 2*10**8
	n135 = 0
	E135 = 62790-4186
	A136 = 2*10**8
	n136 = 0
	E136 = 54418-4186
	A137 = 2*10**8
	n137 = 0
	E137 = 54418-4186
	A138 = 2*10**8.5
	n138 = 0
	E138 = 41860-4186
	A139 = 2*10**8
	n139 = 0
	E139 = 54418-4186
	A140 = 2*10**8
	n140 = 0
	E140 = 54418-4186
	A141 = 2*10**8.5
	n141 = 0
	E141 = 48139-4186
	A142 = 2*10**8
	n142 = 0
	E142 = 54418-4186
	A143 = 2*10**8.5
	n143 = 0
	E143 = 54418-4186
	A144 = 2*10**8
	n144 = 0
	E144 = 46046-4186
	A145 = 2*10**8
	n145 = 0
	E145 = 83720-4186
	A146 = 2*10**8
	n146 = 0
	E146 = 56511-4186
	A147 = 2*10**8
	n147 = 0
	E147 = 56511-4186
	A148 = 2*10**8.5
	n148 = 0
	E148 = 41860-4186
	A149 = 2*10**8
	n149 = 0
	E149 = 54418-4186
	A150 = 2*10**9.5
	n150 = 0
	E150 = 14651-4186
	A151 = 2*10**8.5
	n151 = 0
	E151 = 33069-4186
	A152 = 2*10**8.5
	n152 = 0
	E152 = 54418-4186
	A153 = 2*10**8
	n153 = 0
	E153 = 54418-4186
	A154 = 2*10**8
	n154 = 0
	E154 = 71162-4186
	A155 = 2*10**8
	n155 = 0
	E155 = 54418-4186
	A156 = 2*10**8
	n156 = 0
	E156 = 62790-4186
	A157 = 2*10**8
	n157 = 0
	E157 = 54418-4186
	A158 = 2*10**8
	n158 = 0
	E158 = 54418-4186
	A159 = 2*10**8.5
	n159 = 0
	E159 = 41860-4186
	A160 = 2*10**8
	n160 = 0
	E160 = 54418-4186
	A161 = 2*10**8
	n161 = 0
	E161 = 54418-4186
	A162 = 2*10**8.5
	n162 = 0
	E162 = 48139-4186
	A163 = 2*10**8
	n163 = 0
	E163 = 54418-4186
	A164 = 2*10**8.5
	n164 = 0
	E164 = 54418-4186
	A165 = 2*10**8
	n165 = 0
	E165 = 46046-4186
	A166 = 2*10**8
	n166 = 0
	E166 = 83720-4186
	A167 = 2*10**8
	n167 = 0
	E167 = 56511-4186
	A168 = 2*10**8
	n168 = 0
	E168 = 56511-4186
	A169 = 2*10**8.5
	n169 = 0
	E169 = 41860-4186
	A170 = 2*10**8
	n170 = 0
	E170 = 54418-4186
	A171 = 2*10**9.5
	n171 = 0
	E171 = 14651-4186
	A172 = 2*10**8.5
	n172 = 0
	E172 = 33069-4186
	A173 = 2*10**8.5
	n173 = 0
	E173 = 54418-4186
	A174 = 2*10**8
	n174 = 0
	E174 = 54418-4186
	A175 = 1*10**8
	n175 = 0
	E175 = 71162-4186
	A176 = 1*10**8
	n176 = 0
	E176 = 54418-4186
	A177 = 1*10**8
	n177 = 0
	E177 = 62790-4186
	A178 = 1*10**8
	n178 = 0
	E178 = 54418-4186
	A179 = 1*10**8
	n179 = 0
	E179 = 54418-4186
	A180 = 1*10**8.5
	n180 = 0
	E180 = 41860-4186
	A181 = 1*10**8
	n181 = 0
	E181 = 54418-4186
	A182 = 1*10**8
	n182 = 0
	E182 = 54418-4186
	A183 = 1*10**8.5
	n183 = 0
	E183 = 48139-4186
	A184 = 1*10**8
	n184 = 0
	E184 = 54418-4186
	A185 = 1*10**8.5
	n185 = 0
	E185 = 54418-4186
	A186 = 1*10**8
	n186 = 0
	E186 = 46046-4186
	A187 = 1*10**8
	n187 = 0
	E187 = 83720-4186
	A188 = 1*10**8
	n188 = 0
	E188 = 56511-4186
	A189 = 1*10**8
	n189 = 0
	E189 = 56511-4186
	A190 = 1*10**8.5
	n190 = 0
	E190 = 41860-4186
	A191 = 1*10**8
	n191 = 0
	E191 = 54418-4186
	A192 = 1*10**9.5
	n192 = 0
	E192 = 14651-4186
	A193 = 1*10**8.5
	n193 = 0
	E193 = 33069-4186
	A194 = 1*10**8.5
	n194 = 0
	E194 = 54418-4186
	A195 = 1*10**8
	n195 = 0
	E195 = 54418-4186
	A196 = 1*10**8
	n196 = 0
	E196 = 71162-4186
	A197 = 1*10**8
	n197 = 0
	E197 = 54418-4186
	A198 = 1*10**8
	n198 = 0
	E198 = 62790-4186
	A199 = 1*10**8
	n199 = 0
	E199 = 54418-4186
	A200 = 1*10**8
	n200 = 0
	E200 = 54418-4186
	A201 = 1*10**8.5
	n201 = 0
	E201 = 41860-4186
	A202 = 1*10**8
	n202 = 0
	E202 = 54418-4186
	A203 = 1*10**8
	n203 = 0
	E203 = 54418-4186
	A204 = 1*10**8.5
	n204 = 0
	E204 = 48139-4186
	A205 = 1*10**8
	n205 = 0
	E205 = 54418-4186
	A206 = 1*10**8.5
	n206 = 0
	E206 = 54418-4186
	A207 = 1*10**8
	n207 = 0
	E207 = 46046-4186
	A208 = 1*10**8
	n208 = 0
	E208 = 83720-4186
	A209 = 1*10**8
	n209 = 0
	E209 = 56511-4186
	A210 = 1*10**8
	n210 = 0
	E210 = 56511-4186
	A211 = 1*10**8.5
	n211 = 0
	E211 = 41860-4186
	A212 = 1*10**8
	n212 = 0
	E212 = 54418-4186
	A213 = 1*10**9.5
	n213 = 0
	E213 = 14651-4186
	A214 = 1*10**8.5
	n214 = 0
	E214 = 33069-4186
	A215 = 1*10**8.5
	n215 = 0
	E215 = 54418-4186
	A216 = 1*10**8
	n216 = 0
	E216 = 54418-4186
	A217 = 2*10**8
	n217 = 0
	E217 = 71162-4186
	A218 = 2*10**8
	n218 = 0
	E218 = 54418-4186
	A219 = 2*10**8
	n219 = 0
	E219 = 62790-4186
	A220 = 2*10**8
	n220 = 0
	E220 = 54418-4186
	A221 = 2*10**8
	n221 = 0
	E221 = 54418-4186
	A222 = 2*10**8.5
	n222 = 0
	E222 = 41860-4186
	A223 = 2*10**8
	n223 = 0
	E223 = 54418-4186
	A224 = 2*10**8
	n224 = 0
	E224 = 54418-4186
	A225 = 2*10**8.5
	n225 = 0
	E225 = 48139-4186
	A226 = 2*10**8
	n226 = 0
	E226 = 54418-4186
	A227 = 2*10**8.5
	n227 = 0
	E227 = 54418-4186
	A228 = 2*10**8
	n228 = 0
	E228 = 46046-4186
	A229 = 2*10**8
	n229 = 0
	E229 = 83720-4186
	A230 = 2*10**8
	n230 = 0
	E230 = 56511-4186
	A231 = 2*10**8
	n231 = 0
	E231 = 56511-4186
	A232 = 2*10**8.5
	n232 = 0
	E232 = 41860-4186
	A233 = 2*10**8
	n233 = 0
	E233 = 54418-4186
	A234 = 2*10**9.5
	n234 = 0
	E234 = 14651-4186
	A235 = 2*10**8.5
	n235 = 0
	E235 = 33069-4186
	A236 = 2*10**8.5
	n236 = 0
	E236 = 54418-4186
	A237 = 2*10**8
	n237 = 0
	E237 = 54418-4186
	A238 = 1*10**8
	n238 = 0
	E238 = 71162-4186
	A239 = 1*10**8
	n239 = 0
	E239 = 54418-4186
	A240 = 1*10**8
	n240 = 0
	E240 = 62790-4186
	A241 = 1*10**8
	n241 = 0
	E241 = 54418-4186
	A242 = 1*10**8
	n242 = 0
	E242 = 54418-4186
	A243 = 1*10**8.5
	n243 = 0
	E243 = 41860-4186
	A244 = 1*10**8
	n244 = 0
	E244 = 54418-4186
	A245 = 1*10**8
	n245 = 0
	E245 = 54418-4186
	A246 = 1*10**8.5
	n246 = 0
	E246 = 48139-4186
	A247 = 1*10**8
	n247 = 0
	E247 = 54418-4186
	A248 = 1*10**8.5
	n248 = 0
	E248 = 54418-4186
	A249 = 1*10**8
	n249 = 0
	E249 = 46046-4186
	A250 = 1*10**8
	n250 = 0
	E250 = 83720-4186
	A251 = 1*10**8
	n251 = 0
	E251 = 56511-4186
	A252 = 1*10**8
	n252 = 0
	E252 = 56511-4186
	A253 = 1*10**8.5
	n253 = 0
	E253 = 41860-4186
	A254 = 1*10**8
	n254 = 0
	E254 = 54418-4186
	A255 = 1*10**9.5
	n255 = 0
	E255 = 14651-4186
	A256 = 1*10**8.5
	n256 = 0
	E256 = 33069-4186
	A257 = 1*10**8.5
	n257 = 0
	E257 = 54418-4186
	A258 = 1*10**8
	n258 = 0
	E258 = 54418-4186
	A259 = 1*10**8
	n259 = 0
	E259 = 71162-4186
	A260 = 1*10**8
	n260 = 0
	E260 = 54418-4186
	A261 = 1*10**8
	n261 = 0
	E261 = 62790-4186
	A262 = 1*10**8
	n262 = 0
	E262 = 54418-4186
	A263 = 1*10**8
	n263 = 0
	E263 = 54418-4186
	A264 = 1*10**8.5
	n264 = 0
	E264 = 41860-4186
	A265 = 1*10**8
	n265 = 0
	E265 = 54418-4186
	A266 = 1*10**8
	n266 = 0
	E266 = 54418-4186
	A267 = 1*10**8.5
	n267 = 0
	E267 = 48139-4186
	A268 = 1*10**8
	n268 = 0
	E268 = 54418-4186
	A269 = 1*10**8.5
	n269 = 0
	E269 = 54418-4186
	A270 = 1*10**8
	n270 = 0
	E270 = 46046-4186
	A271 = 1*10**8
	n271 = 0
	E271 = 83720-4186
	A272 = 1*10**8
	n272 = 0
	E272 = 56511-4186
	A273 = 1*10**8
	n273 = 0
	E273 = 56511-4186
	A274 = 1*10**8.5
	n274 = 0
	E274 = 41860-4186
	A275 = 1*10**8
	n275 = 0
	E275 = 54418-4186
	A276 = 1*10**9.5
	n276 = 0
	E276 = 14651-4186
	A277 = 1*10**8.5
	n277 = 0
	E277 = 33069-4186
	A278 = 1*10**8.5
	n278 = 0
	E278 = 54418-4186
	A279 = 1*10**8
	n279 = 0
	E279 = 54418-4186
	A280 = 1*10**8
	n280 = 0
	E280 = 71162--20930
	A281 = 1*10**8
	n281 = 0
	E281 = 54418--20930
	A282 = 1*10**8
	n282 = 0
	E282 = 62790--20930
	A283 = 1*10**8
	n283 = 0
	E283 = 54418--20930
	A284 = 1*10**8
	n284 = 0
	E284 = 54418--20930
	A285 = 1*10**8.5
	n285 = 0
	E285 = 41860--20930
	A286 = 1*10**8
	n286 = 0
	E286 = 54418--20930
	A287 = 1*10**8
	n287 = 0
	E287 = 54418--20930
	A288 = 1*10**8.5
	n288 = 0
	E288 = 48139--20930
	A289 = 1*10**8
	n289 = 0
	E289 = 54418--20930
	A290 = 1*10**8.5
	n290 = 0
	E290 = 54418--20930
	A291 = 1*10**8
	n291 = 0
	E291 = 46046--20930
	A292 = 1*10**8
	n292 = 0
	E292 = 83720--20930
	A293 = 1*10**8
	n293 = 0
	E293 = 56511--20930
	A294 = 1*10**8
	n294 = 0
	E294 = 56511--20930
	A295 = 1*10**8.5
	n295 = 0
	E295 = 41860--20930
	A296 = 1*10**8
	n296 = 0
	E296 = 54418--20930
	A297 = 1*10**9.5
	n297 = 0
	E297 = 14651--20930
	A298 = 1*10**8.5
	n298 = 0
	E298 = 33069--20930
	A299 = 1*10**8.5
	n299 = 0
	E299 = 54418--20930
	A300 = 1*10**8
	n300 = 0
	E300 = 54418--20930
	A301 = 1*10**8
	n301 = 0
	E301 = 71162-4186
	A302 = 1*10**8
	n302 = 0
	E302 = 54418-4186
	A303 = 1*10**8
	n303 = 0
	E303 = 62790-4186
	A304 = 1*10**8
	n304 = 0
	E304 = 54418-4186
	A305 = 1*10**8
	n305 = 0
	E305 = 54418-4186
	A306 = 1*10**8.5
	n306 = 0
	E306 = 41860-4186
	A307 = 1*10**8
	n307 = 0
	E307 = 54418-4186
	A308 = 1*10**8
	n308 = 0
	E308 = 54418-4186
	A309 = 1*10**8.5
	n309 = 0
	E309 = 48139-4186
	A310 = 1*10**8
	n310 = 0
	E310 = 54418-4186
	A311 = 1*10**8.5
	n311 = 0
	E311 = 54418-4186
	A312 = 1*10**8
	n312 = 0
	E312 = 46046-4186
	A313 = 1*10**8
	n313 = 0
	E313 = 83720-4186
	A314 = 1*10**8
	n314 = 0
	E314 = 56511-4186
	A315 = 1*10**8
	n315 = 0
	E315 = 56511-4186
	A316 = 1*10**8.5
	n316 = 0
	E316 = 41860-4186
	A317 = 1*10**8
	n317 = 0
	E317 = 54418-4186
	A318 = 1*10**9.5
	n318 = 0
	E318 = 14651-4186
	A319 = 1*10**8.5
	n319 = 0
	E319 = 33069-4186
	A320 = 1*10**8.5
	n320 = 0
	E320 = 54418-4186
	A321 = 1*10**8
	n321 = 0
	E321 = 54418-4186
	A322 = 1*10**8
	n322 = 0
	E322 = 71162-4186
	A323 = 1*10**8
	n323 = 0
	E323 = 54418-4186
	A324 = 1*10**8
	n324 = 0
	E324 = 62790-4186
	A325 = 1*10**8
	n325 = 0
	E325 = 54418-4186
	A326 = 1*10**8
	n326 = 0
	E326 = 54418-4186
	A327 = 1*10**8.5
	n327 = 0
	E327 = 41860-4186
	A328 = 1*10**8
	n328 = 0
	E328 = 54418-4186
	A329 = 1*10**8
	n329 = 0
	E329 = 54418-4186
	A330 = 1*10**8.5
	n330 = 0
	E330 = 48139-4186
	A331 = 1*10**8
	n331 = 0
	E331 = 54418-4186
	A332 = 1*10**8.5
	n332 = 0
	E332 = 54418-4186
	A333 = 1*10**8
	n333 = 0
	E333 = 46046-4186
	A334 = 1*10**8
	n334 = 0
	E334 = 83720-4186
	A335 = 1*10**8
	n335 = 0
	E335 = 56511-4186
	A336 = 1*10**8
	n336 = 0
	E336 = 56511-4186
	A337 = 1*10**8.5
	n337 = 0
	E337 = 41860-4186
	A338 = 1*10**8
	n338 = 0
	E338 = 54418-4186
	A339 = 1*10**9.5
	n339 = 0
	E339 = 14651-4186
	A340 = 1*10**8.5
	n340 = 0
	E340 = 33069-4186
	A341 = 1*10**8.5
	n341 = 0
	E341 = 54418-4186
	A342 = 1*10**8
	n342 = 0
	E342 = 54418-4186
	A343 = 2*10**8
	n343 = 0
	E343 = 71162-4186
	A344 = 2*10**8
	n344 = 0
	E344 = 54418-4186
	A345 = 2*10**8
	n345 = 0
	E345 = 62790-4186
	A346 = 2*10**8
	n346 = 0
	E346 = 54418-4186
	A347 = 2*10**8
	n347 = 0
	E347 = 54418-4186
	A348 = 2*10**8.5
	n348 = 0
	E348 = 41860-4186
	A349 = 2*10**8
	n349 = 0
	E349 = 54418-4186
	A350 = 2*10**8
	n350 = 0
	E350 = 54418-4186
	A351 = 2*10**8.5
	n351 = 0
	E351 = 48139-4186
	A352 = 2*10**8
	n352 = 0
	E352 = 54418-4186
	A353 = 2*10**8.5
	n353 = 0
	E353 = 54418-4186
	A354 = 2*10**8
	n354 = 0
	E354 = 46046-4186
	A355 = 2*10**8
	n355 = 0
	E355 = 83720-4186
	A356 = 2*10**8
	n356 = 0
	E356 = 56511-4186
	A357 = 2*10**8
	n357 = 0
	E357 = 56511-4186
	A358 = 2*10**8.5
	n358 = 0
	E358 = 41860-4186
	A359 = 2*10**8
	n359 = 0
	E359 = 54418-4186
	A360 = 2*10**9.5
	n360 = 0
	E360 = 14651-4186
	A361 = 2*10**8.5
	n361 = 0
	E361 = 33069-4186
	A362 = 2*10**8.5
	n362 = 0
	E362 = 54418-4186
	A363 = 2*10**8
	n363 = 0
	E363 = 54418-4186
	A364 = 1*10**8
	n364 = 0
	E364 = 71162-4186
	A365 = 1*10**8
	n365 = 0
	E365 = 54418-4186
	A366 = 1*10**8
	n366 = 0
	E366 = 62790-4186
	A367 = 1*10**8
	n367 = 0
	E367 = 54418-4186
	A368 = 1*10**8
	n368 = 0
	E368 = 54418-4186
	A369 = 1*10**8.5
	n369 = 0
	E369 = 41860-4186
	A370 = 1*10**8
	n370 = 0
	E370 = 54418-4186
	A371 = 1*10**8
	n371 = 0
	E371 = 54418-4186
	A372 = 1*10**8.5
	n372 = 0
	E372 = 48139-4186
	A373 = 1*10**8
	n373 = 0
	E373 = 54418-4186
	A374 = 1*10**8.5
	n374 = 0
	E374 = 54418-4186
	A375 = 1*10**8
	n375 = 0
	E375 = 46046-4186
	A376 = 1*10**8
	n376 = 0
	E376 = 83720-4186
	A377 = 1*10**8
	n377 = 0
	E377 = 56511-4186
	A378 = 1*10**8
	n378 = 0
	E378 = 56511-4186
	A379 = 1*10**8.5
	n379 = 0
	E379 = 41860-4186
	A380 = 1*10**8
	n380 = 0
	E380 = 54418-4186
	A381 = 1*10**9.5
	n381 = 0
	E381 = 14651-4186
	A382 = 1*10**8.5
	n382 = 0
	E382 = 33069-4186
	A383 = 1*10**8.5
	n383 = 0
	E383 = 54418-4186
	A384 = 1*10**8
	n384 = 0
	E384 = 54418-4186
	A385 = 1*10**8
	n385 = 0
	E385 = 71162-4186
	A386 = 1*10**8
	n386 = 0
	E386 = 54418-4186
	A387 = 1*10**8
	n387 = 0
	E387 = 62790-4186
	A388 = 1*10**8
	n388 = 0
	E388 = 54418-4186
	A389 = 1*10**8
	n389 = 0
	E389 = 54418-4186
	A390 = 1*10**8.5
	n390 = 0
	E390 = 41860-4186
	A391 = 1*10**8
	n391 = 0
	E391 = 54418-4186
	A392 = 1*10**8
	n392 = 0
	E392 = 54418-4186
	A393 = 1*10**8.5
	n393 = 0
	E393 = 48139-4186
	A394 = 1*10**8
	n394 = 0
	E394 = 54418-4186
	A395 = 1*10**8.5
	n395 = 0
	E395 = 54418-4186
	A396 = 1*10**8
	n396 = 0
	E396 = 46046-4186
	A397 = 1*10**8
	n397 = 0
	E397 = 83720-4186
	A398 = 1*10**8
	n398 = 0
	E398 = 56511-4186
	A399 = 1*10**8
	n399 = 0
	E399 = 56511-4186
	A400 = 1*10**8.5
	n400 = 0
	E400 = 41860-4186
	A401 = 1*10**8
	n401 = 0
	E401 = 54418-4186
	A402 = 1*10**9.5
	n402 = 0
	E402 = 14651-4186
	A403 = 1*10**8.5
	n403 = 0
	E403 = 33069-4186
	A404 = 1*10**8.5
	n404 = 0
	E404 = 54418-4186
	A405 = 1*10**8
	n405 = 0
	E405 = 54418-4186


	# Initial conditions	
	ADIO = ADIOM2 = ALD3 = C10H2 = C10H2M2 = C10H2M4 = C2H6 = C3H4O = C3H4O2 = C3H6 = C3H6O2 = C3H8O2 = CH2CO = CH3CHO = CH3OH = CH4 = CHAR = CO = CO2 = COUMARYL = ETOH = H2 = H2O = KET = KETD = KETDM2 = KETM2 = LIG = LIGC = LIGH = LIGM2 = LIGO = MGUAI = OH = PADIO = PADIOM2 = PC2H2 = PCH2OH = PCH2P = PCH3 = PCHO = PCHOHP = PCHP2 = PCOH = PCOHP2 = PCOS = PFET3 = PFET3M2 = PH2 = PHENOL = PKETM2 = PLIG = PLIGM2 = PRADIO = PRADIOM2 = PRFET3 = PRFET3M2 = PRKETM2 = PRLIGH = PRLIGH2 = PRLIGM2A = RADIO = RADIOM2 = RC3H3O = RC3H5O2 = RC3H7O2 = RCH3 = RCH3O = RKET = RKETM2 = RLIGA = RLIGB = RLIGH = RLIGM2A = RLIGM2B = RMGUAI = RPHENOL = RPHENOX = RPHENOXM2 = SYNAPYL = VADIO = VADIOM2 = VCOUMARYL = VKET = VKETD = VKETDM2 = VKETM2 = VMGUAI = VPHENOL = VSYNAPYL = 0
	# ODE solver parameters
	abserr = 1e-11
	relerr = 1e-09
	numpoints = int(np.ceil(stoptime))+1
        
	t = [stoptime * int(i) / (numpoints - 1) for i in range(numpoints)]
	y0 = [T0, ADIO, ADIOM2, ALD3, C10H2, C10H2M2, C10H2M4, C2H6, C3H4O, C3H4O2, C3H6, C3H6O2, C3H8O2, CH2CO, CH3CHO, CH3OH, CH4, CHAR, CO, CO2, COUMARYL, ETOH, H2, H2O, KET, KETD, KETDM2, KETM2, LIG, LIGC, LIGH, LIGM2, LIGO, MGUAI, OH, PADIO, PADIOM2, PC2H2, PCH2OH, PCH2P, PCH3, PCHO, PCHOHP, PCHP2, PCOH, PCOHP2, PCOS, PFET3, PFET3M2, PH2, PHENOL, PKETM2, PLIG, PLIGC, PLIGH, PLIGM2, PLIGO, PRADIO, PRADIOM2, PRFET3, PRFET3M2, PRKETM2, PRLIGH, PRLIGH2, PRLIGM2A, RADIO, RADIOM2, RC3H3O, RC3H5O2, RC3H7O2, RCH3, RCH3O, RKET, RKETM2, RLIGA, RLIGB, RLIGH, RLIGM2A, RLIGM2B, RMGUAI, RPHENOL, RPHENOX, RPHENOXM2, SYNAPYL, VADIO, VADIOM2, VCOUMARYL, VKET, VKETD, VKETDM2, VKETM2, VMGUAI, VPHENOL, VSYNAPYL]
	p = [alpha, R, A0, n0, E0, A1, n1, E1, A2, n2, E2, A3, n3, E3, A4, n4, E4, A5, n5, E5, A6, n6, E6, A7, n7, E7, A8, n8, E8, A9, n9, E9, A10, n10, E10, A11, n11, E11, A12, n12, E12, A13, n13, E13, A14, n14, E14, A15, n15, E15, A16, n16, E16, A17, n17, E17, A18, n18, E18, A19, n19, E19, A20, n20, E20, A21, n21, E21, A22, n22, E22, A23, n23, E23, A24, n24, E24, A25, n25, E25, A26, n26, E26, A27, n27, E27, A28, n28, E28, A29, n29, E29, A30, n30, E30, A31, n31, E31, A32, n32, E32, A33, n33, E33, A34, n34, E34, A35, n35, E35, A36, n36, E36, A37, n37, E37, A38, n38, E38, A39, n39, E39, A40, n40, E40, A41, n41, E41, A42, n42, E42, A43, n43, E43, A44, n44, E44, A45, n45, E45, A46, n46, E46, A47, n47, E47, A48, n48, E48, A49, n49, E49, A50, n50, E50, A51, n51, E51, A52, n52, E52, A53, n53, E53, A54, n54, E54, A55, n55, E55, A56, n56, E56, A57, n57, E57, A58, n58, E58, A59, n59, E59, A60, n60, E60, A61, n61, E61, A62, n62, E62, A63, n63, E63, A64, n64, E64, A65, n65, E65, A66, n66, E66, A67, n67, E67, A68, n68, E68, A69, n69, E69, A70, n70, E70, A71, n71, E71, A72, n72, E72, A73, n73, E73, A74, n74, E74, A75, n75, E75, A76, n76, E76, A77, n77, E77, A78, n78, E78, A79, n79, E79, A80, n80, E80, A81, n81, E81, A82, n82, E82, A83, n83, E83, A84, n84, E84, A85, n85, E85, A86, n86, E86, A87, n87, E87, A88, n88, E88, A89, n89, E89, A90, n90, E90, A91, n91, E91, A92, n92, E92, A93, n93, E93, A94, n94, E94, A95, n95, E95, A96, n96, E96, A97, n97, E97, A98, n98, E98, A99, n99, E99, A100, n100, E100, A101, n101, E101, A102, n102, E102, A103, n103, E103, A104, n104, E104, A105, n105, E105, A106, n106, E106, A107, n107, E107, A108, n108, E108, A109, n109, E109, A110, n110, E110, A111, n111, E111, A112, n112, E112, A113, n113, E113, A114, n114, E114, A115, n115, E115, A116, n116, E116, A117, n117, E117, A118, n118, E118, A119, n119, E119, A120, n120, E120, A121, n121, E121, A122, n122, E122, A123, n123, E123, A124, n124, E124, A125, n125, E125, A126, n126, E126, A127, n127, E127, A128, n128, E128, A129, n129, E129, A130, n130, E130, A131, n131, E131, A132, n132, E132, A133, n133, E133, A134, n134, E134, A135, n135, E135, A136, n136, E136, A137, n137, E137, A138, n138, E138, A139, n139, E139, A140, n140, E140, A141, n141, E141, A142, n142, E142, A143, n143, E143, A144, n144, E144, A145, n145, E145, A146, n146, E146, A147, n147, E147, A148, n148, E148, A149, n149, E149, A150, n150, E150, A151, n151, E151, A152, n152, E152, A153, n153, E153, A154, n154, E154, A155, n155, E155, A156, n156, E156, A157, n157, E157, A158, n158, E158, A159, n159, E159, A160, n160, E160, A161, n161, E161, A162, n162, E162, A163, n163, E163, A164, n164, E164, A165, n165, E165, A166, n166, E166, A167, n167, E167, A168, n168, E168, A169, n169, E169, A170, n170, E170, A171, n171, E171, A172, n172, E172, A173, n173, E173, A174, n174, E174, A175, n175, E175, A176, n176, E176, A177, n177, E177, A178, n178, E178, A179, n179, E179, A180, n180, E180, A181, n181, E181, A182, n182, E182, A183, n183, E183, A184, n184, E184, A185, n185, E185, A186, n186, E186, A187, n187, E187, A188, n188, E188, A189, n189, E189, A190, n190, E190, A191, n191, E191, A192, n192, E192, A193, n193, E193, A194, n194, E194, A195, n195, E195, A196, n196, E196, A197, n197, E197, A198, n198, E198, A199, n199, E199, A200, n200, E200, A201, n201, E201, A202, n202, E202, A203, n203, E203, A204, n204, E204, A205, n205, E205, A206, n206, E206, A207, n207, E207, A208, n208, E208, A209, n209, E209, A210, n210, E210, A211, n211, E211, A212, n212, E212, A213, n213, E213, A214, n214, E214, A215, n215, E215, A216, n216, E216, A217, n217, E217, A218, n218, E218, A219, n219, E219, A220, n220, E220, A221, n221, E221, A222, n222, E222, A223, n223, E223, A224, n224, E224, A225, n225, E225, A226, n226, E226, A227, n227, E227, A228, n228, E228, A229, n229, E229, A230, n230, E230, A231, n231, E231, A232, n232, E232, A233, n233, E233, A234, n234, E234, A235, n235, E235, A236, n236, E236, A237, n237, E237, A238, n238, E238, A239, n239, E239, A240, n240, E240, A241, n241, E241, A242, n242, E242, A243, n243, E243, A244, n244, E244, A245, n245, E245, A246, n246, E246, A247, n247, E247, A248, n248, E248, A249, n249, E249, A250, n250, E250, A251, n251, E251, A252, n252, E252, A253, n253, E253, A254, n254, E254, A255, n255, E255, A256, n256, E256, A257, n257, E257, A258, n258, E258, A259, n259, E259, A260, n260, E260, A261, n261, E261, A262, n262, E262, A263, n263, E263, A264, n264, E264, A265, n265, E265, A266, n266, E266, A267, n267, E267, A268, n268, E268, A269, n269, E269, A270, n270, E270, A271, n271, E271, A272, n272, E272, A273, n273, E273, A274, n274, E274, A275, n275, E275, A276, n276, E276, A277, n277, E277, A278, n278, E278, A279, n279, E279, A280, n280, E280, A281, n281, E281, A282, n282, E282, A283, n283, E283, A284, n284, E284, A285, n285, E285, A286, n286, E286, A287, n287, E287, A288, n288, E288, A289, n289, E289, A290, n290, E290, A291, n291, E291, A292, n292, E292, A293, n293, E293, A294, n294, E294, A295, n295, E295, A296, n296, E296, A297, n297, E297, A298, n298, E298, A299, n299, E299, A300, n300, E300, A301, n301, E301, A302, n302, E302, A303, n303, E303, A304, n304, E304, A305, n305, E305, A306, n306, E306, A307, n307, E307, A308, n308, E308, A309, n309, E309, A310, n310, E310, A311, n311, E311, A312, n312, E312, A313, n313, E313, A314, n314, E314, A315, n315, E315, A316, n316, E316, A317, n317, E317, A318, n318, E318, A319, n319, E319, A320, n320, E320, A321, n321, E321, A322, n322, E322, A323, n323, E323, A324, n324, E324, A325, n325, E325, A326, n326, E326, A327, n327, E327, A328, n328, E328, A329, n329, E329, A330, n330, E330, A331, n331, E331, A332, n332, E332, A333, n333, E333, A334, n334, E334, A335, n335, E335, A336, n336, E336, A337, n337, E337, A338, n338, E338, A339, n339, E339, A340, n340, E340, A341, n341, E341, A342, n342, E342, A343, n343, E343, A344, n344, E344, A345, n345, E345, A346, n346, E346, A347, n347, E347, A348, n348, E348, A349, n349, E349, A350, n350, E350, A351, n351, E351, A352, n352, E352, A353, n353, E353, A354, n354, E354, A355, n355, E355, A356, n356, E356, A357, n357, E357, A358, n358, E358, A359, n359, E359, A360, n360, E360, A361, n361, E361, A362, n362, E362, A363, n363, E363, A364, n364, E364, A365, n365, E365, A366, n366, E366, A367, n367, E367, A368, n368, E368, A369, n369, E369, A370, n370, E370, A371, n371, E371, A372, n372, E372, A373, n373, E373, A374, n374, E374, A375, n375, E375, A376, n376, E376, A377, n377, E377, A378, n378, E378, A379, n379, E379, A380, n380, E380, A381, n381, E381, A382, n382, E382, A383, n383, E383, A384, n384, E384, A385, n385, E385, A386, n386, E386, A387, n387, E387, A388, n388, E388, A389, n389, E389, A390, n390, E390, A391, n391, E391, A392, n392, E392, A393, n393, E393, A394, n394, E394, A395, n395, E395, A396, n396, E396, A397, n397, E397, A398, n398, E398, A399, n399, E399, A400, n400, E400, A401, n401, E401, A402, n402, E402, A403, n403, E403, A404, n404, E404, A405, n405, E405]

	ysol = odeint(ODEs, y0, t, args=(p,), atol=abserr, rtol=relerr, mxstep=5000)

	with open('sol_check.dat', 'a+') as f:
		#data_format = '{:15.10f}' * 95
		a = list(zip(t, ysol))
		b = len(a)
		result = str(a[b-1])
		f.write(result)
		#b = list(map(int, a))
		#for tt, yy in a:
			#print(data_format.format(tt, *yy), file=f)
		#print(result)
		#end = time.time()
		#run_time = end - start
		#data_format = '{:15d}' * 95
		#col = list(range(0, 95))
		#print(data_format.format(*col), file=f)
		#print(run_time, file=f)


if __name__ == '__main__':
	run()
