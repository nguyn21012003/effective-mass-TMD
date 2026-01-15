set terminal qt size 700,900

reset

dir = "./mass/"

set datafile separator ","

set key top right font "CMU Serif, 20"
set yrange [0:1.0]
set xrange [0:500]
set ytics 0.05
set xtics 100
set lmargin 15
set rmargin 10
set bmargin 4
set xtics font "CMU Serif,20"
set ytics font "CMU Serif,20"
set xlabel "B(T)" font "CMU Serif,20"
set ylabel 'm_{eff} [m_{0}]' offset -3,0 font "CMU Serif,30"

set style line 81 lt 1 lc rgb "#808080" lw 0.5
set grid back ls 81


set title "K" font "CMU Serif, 20"


plot  dir  . "Mass_TNN_WTe2_GGA.dat"  u  1:2 w lines lw 3 lc rgb "red" title "m_{h} TNN",\
      dir  . "Mass_NN_WTe2_GGA.dat"   u  1:2 w lines dashtype ".." lw 3 lc rgb "red" title "m_{h} NN",\
      dir  . "Mass_TNN_WTe2_GGA.dat"  u  1:4 w lines lw 3 lc rgb "navy" title "m_{e} TNN",\
      dir  . "Mass_NN_WTe2_GGA.dat"   u  1:4 w lines dashtype ".." lw 3 lc rgb "navy" title "m_{e} NN",\
      dir  . "Mass_TNN_WTe2_GGA.dat"  u 1:( $2 * $4 / ($2 + $4) ) w lines lt 2 lw 3 lc rgb "orange" title "m_{r} TNN",\
      dir  . "Mass_NN_WTe2_GGA.dat"   u 1:( $2 * $4 / ($2 + $4) ) w lines lw 3 dashtype ".." lc rgb "orange"  title "m_{r} NN"


