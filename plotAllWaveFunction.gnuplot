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


dir = "./Wed-09-03/TNN/" 
do for [i=84:124] {
    set label sprintf("(%d)", i) at graph 0.05,0.95 front font "Arial,20"
    plot dir . "B30.dat" using 1:(column(i+2)) w l lw 2 lc "#0197f6" notitle
    unset label
}
unset multiplot
