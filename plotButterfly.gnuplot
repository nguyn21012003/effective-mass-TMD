set terminal qt size 420,900

NNdir = "./Wed-08-20/NN/"
TNNdir = "./Thu-08-21/TNN/"
set datafile separator ","
set xrange [0:50]
set yrange [*:1.69]

set key top left font "Arial,20"


set lmargin 13
set bmargin 4
set tics nomirror out
set xtics 25 font "Arial,16"
set ytics 0.01 font "Arial,16"
set mytics 2
set mxtics 5
set xlabel "B(T)" font "Arial,20"
set ylabel "Energy (eV)" offset -2,0 font "Arial,20"
# plot dir . "3band_Lambda2q_dataHofstadterButterfly_q_797_MoS2_GGA_G.dat" u 1:3 with points pt 7 ps 0.3 lc rgb 'black' title "2q"


# plot TNNdir . "3band_Lambda2q_dataHofstadterButterfly_q_297_MoS2_GGA_G.dat" u 1:4 w l lw 5 lc rgb "purple" title "omega_{c}",\
TNNdir . "3band_Lambda2q_dataHofstadterButterfly_q_297_MoS2_GGA_G.dat" u 1:4 with points pt 7 ps 0.3 lc rgb "purple" title "omega_{c}",\
TNNdir . "3band_Lambda2q_dataHofstadterButterfly_q_297_MoS2_GGA_G.dat" u 1:5 w l lw 5 lc rgb "purple" title "omega_{c}",\
TNNdir . "3band_Lambda2q_dataHofstadterButterfly_q_297_MoS2_GGA_G.dat" u 1:5 with points pt 7 ps 0.3 lc rgb "purple" title "omega_{c}",\
TNNdir . "3band_Lambda2q_dataHofstadterButterfly_q_297_MoS2_GGA_G.dat" u 1:6 w l lw 5 lc rgb "purple" title "omega_{c}",\
TNNdir . "3band_Lambda2q_dataHofstadterButterfly_q_297_MoS2_GGA_G.dat" u 1:6 with points pt 7 ps 0.3 lc rgb "purple" title "omega_{c}"

# plot 1.5 with points pt 7 ps 3 lc rgb "#bdbdbd"
plot for [i=4:84] TNNdir . "3band_Lambda2q_dataHofstadterButterfly_q_2001_MoS2_GGA_G.dat" u 2:i with points pt 7 ps 0.5 lc rgb "#bdbdbd" notitle "2q+1" ,\
      TNNdir . "3band_Lambda2q_dataHofstadterButterfly_q_2001_MoS2_GGA_G.dat" u 2:4 w l lw 2 lc rgb "red" notitle,\
      TNNdir . "3band_Lambda2q_dataHofstadterButterfly_q_2001_MoS2_GGA_G.dat" u 2:6 w l lw 2 lc rgb "red" notitle,\
      TNNdir . "3band_Lambda2q_dataHofstadterButterfly_q_2001_MoS2_GGA_G.dat" u 2:10 w l lw 2 lc rgb "red" notitle,\
      TNNdir . "3band_Lambda2q_dataHofstadterButterfly_q_2001_MoS2_GGA_G.dat" u 2:12 w l lw 2 lc rgb "red" notitle,\
      TNNdir . "3band_Lambda2q_dataHofstadterButterfly_q_2001_MoS2_GGA_G.dat" u 2:14 w l lw 2 lc rgb "red" notitle,\
      TNNdir . "3band_Lambda2q_dataHofstadterButterfly_q_2001_MoS2_GGA_G.dat" u 2:16 w l lw 2 lc rgb "red" notitle,\
      TNNdir . "3band_Lambda2q_dataHofstadterButterfly_q_2001_MoS2_GGA_G.dat" u 2:8 w l lw 2 lc rgb "red" notitle,\



  # TNNdir . "3band_Lambda2q_dataHofstadterButterfly_q_297_MoS2_GGA_G.dat" u 1:7 with points pt 7 ps 0.3 lc rgb "red" title "m^{*}_{h}/m_{0} in NN case",\
 # TNNdir . "3band_Lambda2q_dataHofstadterButterfly_q_297_MoS2_GGA_G.dat" u 1:5 with points pt 7 ps 0.3 lc rgb "red" title "m^{*}_{h}/m_{0} in NN case",\
     # TNNdir . "3band_Lambda2q_dataHofstadterButterfly_q_297_MoS2_GGA_G.dat" u 1:8 with lines lw 5 lc rgb "magenta" title "m^{*}_{h}/m_{0} TNN case"
    # NNdir . "3band_Lambda2q_dataHofstadterButterfly_q_797_MoS2_GGA_G.dat" u 2:6 with points pt 6 ps 3 lc rgb "orange" title "m^{*}_{e}/m_{0} in NN case",\
    TNNdir . "3band_Lambda2q_dataHofstadterButterfly_q_797_MoS2_GGA_G.dat" u 2:6 with points pt 2 ps 3 lc rgb "red" title "m^{*}_{e}/m_{0} TNN case",\
    


    # "./Sat-08-16/TNN/3band_Lambda2q_dataHofstadterButterfly_q_797_MoS2_GGA_G.dat" u 2:7 with points pt 4 ps 3 lc rgb "red" title "TNN case",\

   # "./Sat-08-16/NN/3band_Lambda2q_dataHofstadterButterfly_q_797_MoS2_GGA_G.dat" u 2:7 with lines lw 5 lc rgb "red" notitle "NN case",\
    "./Sat-08-16/TNN/3band_Lambda2q_dataHofstadterButterfly_q_797_MoS2_GGA_G.dat" u 2:7 with lines lw 5 lc rgb "red" notitle "TNN case"



  # dir . "3band_Lambda2q_dataHofstadterButterfly_q_797_MoS2_GGA_G.dat" u 1:3 with points pt 7 ps 3 lc rgb 'red' title "2q"

