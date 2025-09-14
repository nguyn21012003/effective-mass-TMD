set terminal qt size 1920,1080
set datafile separator ","

set multiplot layout 6,10 title "Γ point"

# set xlabel "|d_{z^{2}}>" font "Arial,20"
set key top right font "Arial,20" 
# set ylabel "|ψ|^{2} (Arb. unit)" font "Arial,20"

set xtics font "Arial,13"
set ytics font "Arial,13"



unset xtics
# unset xtics
# unset ytics


dir = "./Sat-09-13/NN/"
do for [i = 60:119] {
    set label sprintf("(%d)", i) at graph 0.05,0.95 front font "Arial,20"
    plot dir . "WaveFunction_q_297_MoSe2_GGA.dat" using 1:(column(i+2)) w l lw 2 lc "#0197f6" notitle
    unset label
}
unset multiplot
