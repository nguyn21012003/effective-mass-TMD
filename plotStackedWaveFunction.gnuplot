reset
set terminal qt size 1200,900
set multiplot layout 1,2 rowsfirst
set datafile separator ","



set ylabel "Wavefunction (offset)"
set key left top font "Arial,20"
unset xtics


dir = "./Wed-08-20/NN/"
offset = 0.1  # khoảng dịch theo trục y



set label "(a) 2q" at graph 0.05,0.95 front font "Arial,20"
set key right top
set ylabel "Wavefunction + offset"
plot \
    dir."3band_PlotEigenVectors_q_997_MoS2_GGA_G_vals_vecs.dat" \
        using 1:( $2 + 0*offset ) w l lw 3 lc "purple" title "orbital d_{0}", \
    '' using 1:( $3 + 1*offset ) w l lw 3 lc "red"    title "orbital d_{1}", \
    '' using 1:( $4 + 2*offset ) w l lw 3 lc "blue"   title "orbital d_{2}"
unset label

set label "(b) 2q+1" at graph 0.05,0.95 front font "Arial,20"
set key right top
unset ylabel
plot \
    dir."3band_PlotEigenVectors_q_997_MoS2_GGA_G_vals_vecs.dat" \
        using 1:( $5 + 0*offset ) w l lw 3 lc "purple" title "orbital d_{0}", \
    '' using 1:( $6 + 1*offset ) w l lw 3 lc "red"    title "orbital d_{1}", \
    '' using 1:( $7 + 2*offset ) w l lw 3 lc "blue"   title "orbital d_{2}"
unset label

unset multiplot
