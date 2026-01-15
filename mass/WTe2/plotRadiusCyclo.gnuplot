set terminal qt size 700,900 enhanced


dir = "./radius/"

set datafile separator ","

set key font "CMU Serif, 20"
# set xrange [*:101]
# set yrange [0:100]
set ytics 50
# set xtics 10
set mxtics 2
set mytics 2
set tics out
set tics nomirror
set lmargin 15
set rmargin 10
set bmargin 4
set xtics font "CMU Serif,20"
set ytics font "CMU Serif,20"
set xlabel "B(T)" font "CMU Serif,20"
set ylabel "r_{c}/a" offset -3,0 font "CMU Serif,30"

set grid

set title "K'" font "CMU Serif, 20"


plot  dir  . "Radius_WTe2_TNN.dat"  u  1:3  w lines lw 3  lc rgb "light-red" title "r@^h_c TNN case",\
      dir  . "Radius_WTe2_NN.dat"   u  1:3  w lines lw 3  lc rgb "green" title "r@^h_c NN case",\
      dir  . "Radius_WTe2_TNN.dat"  u  1:5  w lines lw 3  lc rgb "navy" title "r@^e_c TNN case",\
      dir  . "Radius_WTe2_NN.dat"   u  1:5  w lines lw 3  lc rgb "magenta" title "r@^e_c NN case"

