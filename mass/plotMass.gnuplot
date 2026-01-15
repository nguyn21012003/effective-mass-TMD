set terminal qt size 1200,700

unset multiplot
reset

MoS2 = "./MoS2/mass/"
MoSe2 = "./MoSe2/mass"
MoTe2 = "./MoTe2/mass"
WS2  = "./WS2/mass"
WSe2 = "./WSe2/mass"
WTe2 = "./WTe2/mass"

set datafile separator ","

# ============================
# LAYOUT 3 × 2
# ============================
set multiplot

set lmargin 17
set key top right font "CMU Serif, 20"
set yrange [0:1.1]
set xrange [0:500]
set ytics 0.2
set xtics 100
set xtics font "CMU Serif,20"
set ytics font "CMU Serif,20"

set style line 81 lt 1 lc rgb "#808080" lw 0.5
set grid back ls 81
  
# ============================
# PLOT 1 — MoS2
# ============================
set label "(a) MoS_{2}" at 80,0.1 font "CMU Serif,28"
unset xlabel
set format x ""
set ylabel 'm_{eff} [m_{0}]' offset -3,0 font "CMU Serif,30"

dir = MoS2 . "/"
set size 0.34, 0.5
set origin 0.0, 0.5
plot  dir . "Mass_TNN_MoS2_GGA.dat"  u 1:2 w lines lw 3 lc rgb "red"  notitle "m_{h} TNN", \
      dir . "Mass_NN_MoS2_GGA.dat"   u 1:2 w lines dashtype ".." lw 3 lc rgb "red" notitle "m_{h} NN", \
      dir . "Mass_TNN_MoS2_GGA.dat"  u 1:4 w lines lw 3 lc rgb "navy" notitle "m_{e} TNN", \
      dir . "Mass_NN_MoS2_GGA.dat"   u 1:4 w lines dashtype ".." lw 3 lc rgb "navy" notitle "m_{e} NN", \
      dir . "Mass_TNN_MoS2_GGA.dat"  u 1:( $2 * $4 / ($2 + $4) ) w lines lw 3 lc rgb "orange" notitle "m_{r} TNN", \
      dir . "Mass_NN_MoS2_GGA.dat"   u 1:( $2 * $4 / ($2 + $4) ) w lines dashtype ".." lw 3 lc rgb "orange" notitle "m_{r} NN"

# ============================
# PLOT 2 — MoSe2
# ============================
unset label
set label "(b) MoSe_{2}" at 80,0.1 font "CMU Serif,28"
set size 0.34, 0.5
set origin 0.25, 0.5
dir = MoSe2 . "/"
unset ylabel
set format y ""
plot  dir . "Mass_TNN_MoSe2_GGA.dat"  u 1:2 w lines lw 3 lc rgb "red"  notitle "m_{h} TNN", \
      dir . "Mass_NN_MoSe2_GGA.dat"   u 1:2 w lines dashtype ".." lw 3 lc rgb "red" notitle "m_{h} NN", \
      dir . "Mass_TNN_MoSe2_GGA.dat"  u 1:4 w lines lw 3 lc rgb "navy" notitle "m_{e} TNN", \
      dir . "Mass_NN_MoSe2_GGA.dat"   u 1:4 w lines dashtype ".." lw 3 lc rgb "navy" notitle "m_{e} NN", \
      dir . "Mass_TNN_MoSe2_GGA.dat"  u 1:( $2 * $4 / ($2 + $4) ) w lines lw 3 lc rgb "orange" notitle "m_{r} TNN", \
      dir . "Mass_NN_MoSe2_GGA.dat"   u 1:( $2 * $4 / ($2 + $4) ) w lines dashtype ".." lw 3 lc rgb "orange" notitle "m_{r} NN"

# ============================
# PLOT 3 — MoTe2
# ============================
set key at screen 0.99, screen 0.98
unset label
set label "(c) MoTe_{2}" at 80,0.1 font "CMU Serif,28"
set size 0.34, 0.5
set origin 0.5, 0.5
unset ylabel
set format y ""
dir = MoTe2 . "/"
plot  dir . "Mass_TNN_MoTe2_GGA.dat"  u 1:2 w lines lw 3 lc rgb "red" title "m_{h} TNN", \
      dir . "Mass_NN_MoTe2_GGA.dat"   u 1:2 w lines dashtype ".." lw 3 lc rgb "red" title "m_{h} NN", \
      dir . "Mass_TNN_MoTe2_GGA.dat"  u 1:4 w lines lw 3 lc rgb "navy" title "m_{e} TNN", \
      dir . "Mass_NN_MoTe2_GGA.dat"   u 1:4 w lines dashtype ".." lw 3 lc rgb "navy" title "m_{e} NN", \
      dir . "Mass_TNN_MoTe2_GGA.dat"  u 1:( $2 * $4 / ($2 + $4) ) w lines lw 3 lc rgb "orange" title "m_{r} TNN", \
      dir . "Mass_NN_MoTe2_GGA.dat"   u 1:( $2 * $4 / ($2 + $4) ) w lines dashtype ".." lw 3 lc rgb "orange" title "m_{r} NN"

unset key

# ============================
# PLOT 4 — WS2
# ============================
unset label
set xlabel "B(T)" font "CMU Serif,20"
set ylabel 'm_{eff} [m_{0}]' offset -3,0 font "CMU Serif,30"
set format x "%g"
set format y "%g"
set label "(d) WS_{2}" at 80,0.9 font "CMU Serif,28"
set size 0.34, 0.5
set origin 0.0, 0.0
dir = WS2 . "/"
plot  dir . "Mass_TNN_WS2_GGA.dat"  u 1:2 w lines lw 3 lc rgb "red"  notitle "m_{h} TNN", \
      dir . "Mass_NN_WS2_GGA.dat"   u 1:2 w lines dashtype ".." lw 3 lc rgb "red" notitle "m_{h} NN", \
      dir . "Mass_TNN_WS2_GGA.dat"  u 1:4 w lines lw 3 lc rgb "navy" notitle "m_{e} TNN", \
      dir . "Mass_NN_WS2_GGA.dat"   u 1:4 w lines dashtype ".." lw 3 lc rgb "navy" notitle "m_{e} NN", \
      dir . "Mass_TNN_WS2_GGA.dat"  u 1:( $2 * $4 / ($2 + $4) ) w lines lw 3 lc rgb "orange" notitle "m_{r} TNN", \
      dir . "Mass_NN_WS2_GGA.dat"   u 1:( $2 * $4 / ($2 + $4) ) w lines dashtype ".." lw 3 lc rgb "orange" notitle "m_{r} NN"

# ============================
# PLOT 5 — WSe2
# ============================
unset label
set label "(e) WSe_{2}" at 80,0.9 font "CMU Serif,28"
set size 0.34, 0.5
set origin 0.25, 0.0
unset ylabel
set format y ""
dir = WSe2 . "/"
plot  dir . "Mass_TNN_WSe2_GGA.dat"  u 1:2 w lines lw 3 lc rgb "red"  notitle "m_{h} TNN", \
      dir . "Mass_NN_WSe2_GGA.dat"   u 1:2 w lines dashtype ".." lw 3 lc rgb "red" notitle "m_{h} NN", \
      dir . "Mass_TNN_WSe2_GGA.dat"  u 1:4 w lines lw 3 lc rgb "navy" notitle "m_{e} TNN", \
      dir . "Mass_NN_WSe2_GGA.dat"   u 1:4 w lines dashtype ".." lw 3 lc rgb "navy" notitle "m_{e} NN", \
      dir . "Mass_TNN_WSe2_GGA.dat"  u 1:( $2 * $4 / ($2 + $4) ) w lines lw 3 lc rgb "orange" notitle "m_{r} TNN", \
      dir . "Mass_NN_WSe2_GGA.dat"   u 1:( $2 * $4 / ($2 + $4) ) w lines dashtype ".." lw 3 lc rgb "orange" notitle "m_{r} NN"

# ============================
# PLOT 6 — WTe2 (bạn đã làm)
# ============================
unset label
set label "(f) WTe_{2}" at 80,0.9 font "CMU Serif,28"
set size 0.34, 0.5
set origin 0.5, 0.0
unset ylabel
set format y ""
dir = WTe2 . "/"
plot  dir . "Mass_TNN_WTe2_GGA.dat"  u 1:2 w lines lw 3 lc rgb "red"  notitle "m_{h} TNN", \
      dir . "Mass_NN_WTe2_GGA.dat"   u 1:2 w lines dashtype ".." lw 3 lc rgb "red" notitle "m_{h} NN", \
      dir . "Mass_TNN_WTe2_GGA.dat"  u 1:4 w lines lw 3 lc rgb "navy" notitle "m_{e} TNN", \
      dir . "Mass_NN_WTe2_GGA.dat"   u 1:4 w lines dashtype ".." lw 3 lc rgb "navy" notitle "m_{e} NN", \
      dir . "Mass_TNN_WTe2_GGA.dat"  u 1:( $2 * $4 / ($2 + $4) ) w lines lw 3 lc rgb "orange" notitle "m_{r} TNN", \
      dir . "Mass_NN_WTe2_GGA.dat"   u 1:( $2 * $4 / ($2 + $4) ) w lines dashtype ".." lw 3 lc rgb "orange" notitle "m_{r} NN"


unset multiplot
