
set terminal qt size 700,900         
set bmargin at screen 0.07
set datafile separator ","
pi = 3.14159265358979323846
set xtics ("-M" -2 * pi / 3.19, "-K" -4 * pi / (3 * 3.19), "Γ" 0, "K" 4 * pi / (3 * 3.19), "M" 2 * pi / 3.19)
set ytics 1

set key top font "Arial,20"
set xtics font ",20" offset 0,-0.5
set ytics font ",20"
set ylabel 'Energy(eV)' font 'Arial,20'
set xlabel '' font 'Arial,20'
set style line 81 lc rgb "#808080" lw 2.5
set grid xtics ls 81
# set key outside top center 
# set label "(b)" at -1.8,4.7 font "Arial,40"
set yrange [*:5.0]

dir = "./Wed-09-03/NN/" 

plot dir . "eigenvalue.csv" u 1:3 w l lw 4 lc "black" notitle,\
   dir . "eigenvalue.csv" u 1:4 w l lw 4 lc "black" notitle,\
   dir . "eigenvalue.csv" u 1:5 w l lw 4 lc "black" notitle


# plot "eigenvalueTNNup.txt" using 1:3 w l lw 4 lc "red" notitle "spin up" ,\
    "eigenvalueTNNdown.txt" using 1:3  w l lw 4 lc rgb "blue" notitle "spin down",\
    "eigenvalueTNNup.txt" using 1:4  w l lw 4 lc rgb "red"  notitle ,\
    "eigenvalueTNNdown.txt" using 1:4  w l lw 4 lc rgb "blue" notitle ,\
    "eigenvalueTNNup.txt" using 1:5  w l lw 4 lc rgb "red"  notitle ,\
    "eigenvalueTNNdown.txt" using 1:5  w l lw 4 lc rgb "blue" notitle ,\
     1/0 with points pt 7 ps 2 lc rgb 'red' title 'spin up' ,\
     1/0 with points pt 7 ps 2 lc rgb 'blue' title 'spin down'


# plot "hnn_test_1000_psoc.csv" using 1:3 w l lw 4 lc "red" notitle "spin up" ,\
    "hnn_test_1000_msoc.csv" using 1:3  w l lw 4 lc rgb "blue" notitle "spin down",\
    "hnn_test_1000_psoc.csv" using 1:4  w l lw 4 lc rgb "red"  notitle ,\
    "hnn_test_1000_msoc.csv" using 1:4  w l lw 4 lc rgb "blue" notitle ,\
    "hnn_test_1000_psoc.csv" using 1:5  w l lw 4 lc rgb "red"  notitle ,\
    "hnn_test_1000_msoc.csv" using 1:5  w l lw 4 lc rgb "blue" notitle ,\
     1/0 with points pt 7 ps 2 lc rgb 'red' title 'spin up' ,\
     1/0 with points pt 7 ps 2 lc rgb 'blue' title 'spin down'

