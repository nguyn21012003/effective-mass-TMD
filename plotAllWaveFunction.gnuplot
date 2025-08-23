set terminal qt size 1920,1080
set datafile separator ","

set multiplot layout 8,10 title "Γ point"

# set xlabel "|d_{z^{2}}>" font "Arial,20"
set key top right font "Arial,20" 
# set ylabel "|ψ|^{2} (Arb. unit)" font "Arial,20"

set xtics font "Arial,13"
set ytics font "Arial,13"



unset xtics
# unset xtics
# unset ytics


dir = "./Sat-08-23/TNN/" 
do for [i=0:79] {
    set label sprintf("(%d)", i+1) at graph 0.05,0.95 front font "Arial,20"
    plot dir . "3band_PlotEigenVectors_q_4001_MoS2_GGA_G_vals_vecs.dat" using 1:(column(i+2)) w l lw 2 lc "#0197f6" notitle
    unset label
}
unset multiplot
