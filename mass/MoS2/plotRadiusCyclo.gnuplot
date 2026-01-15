set terminal qt size 700,900 enhanced


dir = "./radius/"

set datafile separator ","

set key font "CMU Serif,28"
# set xrange [*:101]
# set yrange [0:100]
set ytics 50
set xtics 100
set mxtics 2
set mytics 2
set tics out
set tics nomirror
set lmargin 15
set rmargin 10
set bmargin 4
set xtics font "CMU Serif,28"
set ytics font "CMU Serif,28"
set xlabel "B(T)" font "CMU Serif,28"
set ylabel "r_{c}/a" offset -3,0 font "CMU Serif,30"

set grid

# set title "K" font "CMU Serif, 20"


plot dir  . "Radius_MoS2_TNN.dat"  u  1:4  w lines lw 3  lc rgb "red" notitle "r@^e_c TNN model",\
       # dir  . "Radius_MoS2_NN.dat"   u  1:4  w lines dashtype ".." lw 3  lc rgb "red" title "r@^e_c NN model",\
       # dir  . "Radius_MoS2_TNN.dat"   u  1:5  w lines lw 3  lc rgb "blue" title "r@^e_c NN model",\
       # dir  . "Radius_MoS2_NN.dat"   u  1:5  w lines dashtype ".." lw 3  lc rgb "red" title "r@^e_c NN model",\
      # dir  . "Radius_MoS2_TNN.dat"  u  1:2  w lines lw 3  lc rgb "light-red" title "r@^h_c TNN model",\
      # dir  . "Radius_MoS2_NN.dat"   u  1:2  w lines lw 3  lc rgb "green" title "r@^h_c NN model",\

