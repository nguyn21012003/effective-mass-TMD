set terminal qt size 700,900

dir = "./mass/"

set datafile separator ","

set key font "CMU Serif, 20"
set yrange [0:1]
set ytics 0.1
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


set title "K'" font "CMU Serif, 20"


plot  dir  . "Mass_TNN_MoS2_GGA.dat"  u 1:( $3 * $4 / ($3 + $4) ) w lines lt 2 lw 3 lc rgb "orange" notitle,\
      dir  . "Mass_NN_MoS2_GGA.dat"   u 1:( $3 * $4 / ($3 + $4) ) w lines lw 3 dashtype ".." lc rgb "orange"  notitle,\
      dir  . "Mass_TNN_MoS2_GGA.dat"  u 1:( $2 * $5 / ($2 + $5) ) w linespoints lt 2 lw 3 lc rgb "orange" notitle,\
      dir  . "Mass_NN_MoS2_GGA.dat"   u 1:( $2 * $5 / ($2 + $5) ) w linespoints lw 3 dashtype ".." lc rgb "orange"  notitle,\
 
 
      # dir  . "Mass_TNN_MoS2_GGA.dat"  u  1:3 w lines lw 3 lc rgb "red" notitle ,\
      # dir  . "Mass_NN_MoS2_GGA.dat"   u  1:3 w lines dashtype ".." lw 3 lc rgb "red" notitle,\
      # dir  . "Mass_TNN_MoS2_GGA.dat"  u  1:5 w lines lw 3 lc rgb "navy" notitle ,\
      # dir  . "Mass_NN_MoS2_GGA.dat"   u  1:5 w lines dashtype ".." lw 3 lc rgb "navy" notitle,\
