reset
set terminal qt size 420,900
set datafile separator ","

set bmargin 4

set yrange [1.58:1.72]
set xrange [250:350]

set xtics 25 font "CMU Serif,20"
set ytics 0.01 font "CMU Serif,20"
set mytics 2
set mxtics 5
unset tics

set xlabel "|0>" offset 0,-2 font "CMU Serif,20"


dir = "./Fri-08-22/NN/"
offset = 1.6  # khoảng dịch theo trục y



plot \
    dir."3band_PlotEigenVectors_q_297_MoS2_GGA_G_vals_vecs.dat" \
        using 1:( $3/5 + 1.5857581221263746)    w l lw 3 lc "blue" notitle ,\
    ''  using 1:( $5/5 + 1.6128998889188293)     w l lw 3 lc "blue"      notitle,\
    ''  using 1:( $6/5 + 1.6308762080661343) w l lw 3 lc "purple" notitle,\
    ''  using 1:( $9/5 + 1.6385878397858133) w l lw 3 lc "blue" notitle,\
    ''  using 1:( $11/5 +1.6561823253449328) w l lw 3 lc "purple" notitle,\
    ''  using 1:( $13/5 +1.6629376027272889) w l lw 3 lc "blue" notitle,\
    ''  using 1:( $14/5 +1.6801735536318174) w l lw 3 lc "purple" notitle,\
    ''  using 1:( $18/5 +1.7029453847293399) w l lw 3 lc "purple" notitle,\




