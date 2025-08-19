NNdir = "./Wed-08-20/NN/"
TNNdir = "./Tue-08-19/TNN/"
set datafile separator ","
# set xrange [0:150]
# set yrange [0:2.0]

set key top left font "Arial,20"

set xlabel "B(T)"

set ylabel "m_{eff}/m_{0}" font "Arial,20"
# plot dir . "3band_Lambda2q_dataHofstadterButterfly_q_797_MoS2_GGA_G.dat" u 1:3 with points pt 7 ps 0.3 lc rgb 'black' title "2q"


# plot TNNdir . "3band_Lambda2q_dataHofstadterButterfly_q_297_MoS2_GGA_G.dat" u 1:4 w l lw 5 lc rgb "purple" title "omega_{c}",\
TNNdir . "3band_Lambda2q_dataHofstadterButterfly_q_297_MoS2_GGA_G.dat" u 1:4 with points pt 7 ps 0.3 lc rgb "purple" title "omega_{c}",\
TNNdir . "3band_Lambda2q_dataHofstadterButterfly_q_297_MoS2_GGA_G.dat" u 1:5 w l lw 5 lc rgb "purple" title "omega_{c}",\
TNNdir . "3band_Lambda2q_dataHofstadterButterfly_q_297_MoS2_GGA_G.dat" u 1:5 with points pt 7 ps 0.3 lc rgb "purple" title "omega_{c}",\
TNNdir . "3band_Lambda2q_dataHofstadterButterfly_q_297_MoS2_GGA_G.dat" u 1:6 w l lw 5 lc rgb "purple" title "omega_{c}",\
TNNdir . "3band_Lambda2q_dataHofstadterButterfly_q_297_MoS2_GGA_G.dat" u 1:6 with points pt 7 ps 0.3 lc rgb "purple" title "omega_{c}"


plot NNdir . "3band_Lambda2q_dataHofstadterButterfly_q_297_MoS2_GGA_G.dat" u 1:3 with points pt 7 ps 0.3 lc rgb "red" title "m^{*}_{h}/m_{0} in TNN case",\
  # TNNdir . "3band_Lambda2q_dataHofstadterButterfly_q_297_MoS2_GGA_G.dat" u 1:7 with points pt 7 ps 0.3 lc rgb "red" title "m^{*}_{h}/m_{0} in NN case",\
 # TNNdir . "3band_Lambda2q_dataHofstadterButterfly_q_297_MoS2_GGA_G.dat" u 1:5 with points pt 7 ps 0.3 lc rgb "red" title "m^{*}_{h}/m_{0} in NN case",\
     # TNNdir . "3band_Lambda2q_dataHofstadterButterfly_q_297_MoS2_GGA_G.dat" u 1:8 with lines lw 5 lc rgb "magenta" title "m^{*}_{h}/m_{0} TNN case"
    # NNdir . "3band_Lambda2q_dataHofstadterButterfly_q_797_MoS2_GGA_G.dat" u 2:6 with points pt 6 ps 3 lc rgb "orange" title "m^{*}_{e}/m_{0} in NN case",\
    TNNdir . "3band_Lambda2q_dataHofstadterButterfly_q_797_MoS2_GGA_G.dat" u 2:6 with points pt 2 ps 3 lc rgb "blue" title "m^{*}_{e}/m_{0} TNN case",\
    


    # "./Sat-08-16/TNN/3band_Lambda2q_dataHofstadterButterfly_q_797_MoS2_GGA_G.dat" u 2:7 with points pt 4 ps 3 lc rgb "blue" title "TNN case",\

   # "./Sat-08-16/NN/3band_Lambda2q_dataHofstadterButterfly_q_797_MoS2_GGA_G.dat" u 2:7 with lines lw 5 lc rgb "red" notitle "NN case",\
    "./Sat-08-16/TNN/3band_Lambda2q_dataHofstadterButterfly_q_797_MoS2_GGA_G.dat" u 2:7 with lines lw 5 lc rgb "blue" notitle "TNN case"



  # dir . "3band_Lambda2q_dataHofstadterButterfly_q_797_MoS2_GGA_G.dat" u 1:3 with points pt 7 ps 3 lc rgb 'red' title "2q"

