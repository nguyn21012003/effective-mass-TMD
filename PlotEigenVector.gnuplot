set terminal qt size 1200,900
set datafile separator ","

# set multiplot layout 2,3 title "Γ point"

# set xlabel "|d_{z^{2}}>" font "Arial,20"
set key top right font "Arial,20" 
# set ylabel "|ψ|^{2} (Arb. unit)" font "Arial,20"

set xtics font "Arial,13"
set ytics font "Arial,13"



set xtics
# unset ytics

dir = "./Mon-09-08/NN/" 


# set yrange [0:0.001]
# set xrange [100:300]

#------- (a) --------
set label "(a)" at graph 0.05,0.95 front font "Arial,20"
plot dir . "WaveFunction_q_297_MoSe2_GGA.dat" \
     using 1:63 w l lw 5 lc "#075d37" title "orbital d_{0} landau level n"
unset label

# # -------- (b) --------
# set label "(b)" at graph 0.05,0.95 front font "Arial,20"
# plot dir . "3band_PlotEigenVectors_q_297_MoS2_GGA_G_vals_vecs.dat" \
#      using 1:43 w l lw 5 lc "#e697f3" title "orbital d_{-2} landau level n"
# unset label
#
# # -------- (c) --------
# set label "(c)" at graph 0.05,0.95 front font "Arial,20"
# plot dir . "3band_PlotEigenVectors_q_297_MoS2_GGA_G_vals_vecs.dat" \
#      using 1:84 w l lw 5 lc "#0197f6" title "orbital d_{2} landau level n"
# unset label
#
# # -------- (d) --------
# set label "(d)" at graph 0.05,0.95 front font "Arial,20"
# plot dir . "3band_PlotEigenVectors_q_297_MoS2_GGA_G_vals_vecs.dat" \
#      using 1:3 w l lw 5 lc "#075d37" title "orbital d_{0} landau level n+1"
# unset label
#
# # -------- (e) --------
# set label "(e)" at graph 0.05,0.95 front font "Arial,20"
# plot dir . "3band_PlotEigenVectors_q_297_MoS2_GGA_G_vals_vecs.dat" \
#      using 1:44 w l lw 5 lc "#e697f3" title "orbital d_{-2} landau level n+1"
# unset label
#
# # -------- (f) --------
# set label "(f)" at graph 0.05,0.95 front font "Arial,20"
# plot dir . "3band_PlotEigenVectors_q_297_MoS2_GGA_G_vals_vecs.dat" \
#      using 1:85 w l lw 5 lc "#0197f6" title "orbital d_{2} landau level n+1"
# unset label
#

# unset multiplot
