set terminal qt size 700,900 enhanced

set multiplot 


NNdir = "./Sat-09-06/NN/"
TNNdir = "./Wed-09-03/TNN/"
set datafile separator ","
set xrange [10:80]
set yrange [-0.14:-0.05]

set key font "CMU Serif, 20"


set lmargin 14
set rmargin 13
set bmargin 4
set tics nomirror out
set xtics 20 font "CMU Serif,20"
set ytics font "CMU Serif,20"
set mytics 2
set mxtics 2
set xlabel "B(T)" font "CMU Serif,20"
set ylabel "Energy (eV)" offset -3,0 font "CMU Serif,20"



# plot for [i=2:200] NNdir . "But_q_297_MoS2_GGA.dat" u 1:i with points pt 7 ps 0.5 lc rgb "#bdbdbd" notitle ,\
  NNdir . "But_q_297_MoS2_GGA.dat" u 1:2  with lines lw 2  lc rgb "#1a70bb" notitle "|0,0>_{K}" at end ,\
  NNdir . "But_q_297_MoS2_GGA.dat" u 1:3  with lines lw 2  lc rgb "#1a70bb" notitle "4" at end ,\
  NNdir . "But_q_297_MoS2_GGA.dat" u 1:4  with lines lw 2  lc rgb "#ea801c" notitle "|0,0>_{K'}" at end ,\
  NNdir . "But_q_297_MoS2_GGA.dat" u 1:5  with lines lw 2  lc rgb "#1a70bb" notitle "|0,2>_{K}" at end ,\
  NNdir . "But_q_297_MoS2_GGA.dat" u 1:6 with lines lw 2  lc rgb "#ea801c" notitle "|0,1>_{K'}" at end ,\
  NNdir . "But_q_297_MoS2_GGA.dat" u 1:7 with lines lw 2  lc rgb "#1a70bb" notitle "|0,3>_{K}" at end ,\
  NNdir . "But_q_297_MoS2_GGA.dat" u 1:8 with lines lw 2  lc rgb "#ea801c" notitle "|0,2>_{K'}" at end,\
  NNdir . "But_q_297_MoS2_GGA.dat" u 1:9 with lines lw 2  lc rgb "#ea801c" notitle "|0,2>_{K'}" at end,\
  NNdir . "But_q_297_MoS2_GGA.dat" u 1:10 with lines lw 2  lc rgb "#ea801c" notitle "|0,2>_{K'}" at end,\
  NNdir . "But_q_297_MoS2_GGA.dat" u 1:11 with lines lw 2  lc rgb "#ea801c" notitle "|0,2>_{K'}" at end,\
  NNdir . "But_q_297_MoS2_GGA.dat" u 1:12 with lines lw 2  lc rgb "#ea801c" notitle "|0,2>_{K'}" at end,\
  NNdir . "But_q_297_MoS2_GGA.dat" u 1:13 with lines lw 2  lc rgb "#ea801c" notitle "|0,2>_{K'}" at end




set label " |2,0>_{K}" at 80,-0.05711536078879339 tc rgb "#1a70bb" font "CMU Serif, 20"
set label " |2,1>_{K}" at 80,-0.07299133014948526 tc rgb "#1a70bb" font "CMU Serif, 20"
set label " |2,2>_{K}" at 80,-0.08704535880988144 tc rgb "#1a70bb" font "CMU Serif, 20"
set label " |2,3>_{K}" at 80,-0.10011037750249099 tc rgb "#1a70bb" font "CMU Serif, 20"

set label " |2,0>_{K'}" at 80,-0.08382356967530796 tc rgb "#ea801c" font "CMU Serif, 20"
set label " |2,1>_{K'}" at 80,-0.09752809872044002 tc rgb "#ea801c" font "CMU Serif, 20"
set label " |2,2>_{K'}" at 80,-0.11144864768509424 tc rgb "#ea801c" font "CMU Serif, 20"
set label " |2,3>_{K'}" at 80,-0.12391433639860834 tc rgb "#ea801c" font "CMU Serif, 20"

set label " |2,0>_{Γ}" at 80,-0.061184980569793646 tc rgb "#98c127" font "CMU Serif, 20"
set label " |2,1>_{Γ}" at 80,-0.06568314172560115 tc rgb  "#98c127" font "CMU Serif, 20"
set label " |2,2>_{Γ}" at 80,-0.07072646598213989 tc rgb  "#98c127" font "CMU Serif, 20"
set label " |2,3>_{Γ}" at 80,-0.07631587635660425 tc rgb  "#98c127" font "CMU Serif, 20"

set size 0.5,1
set origin 0.0,0.0

plot for [i=2:200] NNdir . "But_q_297_MoS2_GGA.dat" u 1:i with points pt 7 ps 0.5 lc rgb "#bdbdbd" notitle ,\
  NNdir . "But_q_297_MoS2_GGA.dat" u 1:2  with lines lw 2  lc rgb "#1a70bb" notitle,\
  NNdir . "But_q_297_MoS2_GGA.dat" u 1:3  with lines lw 2  lc rgb "#1a70bb" notitle,\
  NNdir . "But_q_297_MoS2_GGA.dat" u 1:4  with lines lw 2  lc rgb "#1a70bb" notitle,\
  NNdir . "But_q_297_MoS2_GGA.dat" u 1:5  with lines lw 2  lc rgb "#1a70bb" notitle,\
  NNdir . "But_q_297_MoS2_GGA.dat" u 1:6 with lines lw 2  lc rgb "#ea801c" notitle,\
  NNdir . "But_q_297_MoS2_GGA.dat" u 1:7 with lines lw 2  lc rgb "#ea801c" notitle,\
  NNdir . "But_q_297_MoS2_GGA.dat" u 1:8 with lines lw 2  lc rgb "#ea801c" notitle,\
  NNdir . "But_q_297_MoS2_GGA.dat" u 1:9 with lines lw 2  lc rgb "#ea801c" notitle,\
  NNdir . "But_q_297_MoS2_GGA.dat" u 1:10 with lines lw 2  lc rgb "#98c127" notitle,\
  NNdir . "But_q_297_MoS2_GGA.dat" u 1:11 with lines lw 2  lc rgb "#98c127" notitle,\
  NNdir . "But_q_297_MoS2_GGA.dat" u 1:12 with lines lw 2  lc rgb "#98c127" notitle,\
  NNdir . "But_q_297_MoS2_GGA.dat" u 1:13 with lines lw 2  lc rgb "#98c127" notitle

unset label
unset xlabel
unset ylabel
unset tics
unset title
set xrange [350:600]
set size 0.45,1
set origin 0.45,0.0
set xlabel "|0>" offset 0,-2 font "CMU Serif,20"
set ylabel "|{/Symbol y}|^{2}" font "CMU Serif,20"

plot \
    NNdir . "B100.dat" \
        using 1:( $3/8 - 0.058584980569793646)    w l lw 3 lc "#1a70bb" notitle ,\
    ''  using 1:( $5/8 - 0.061184980569793646)     w l lw 3 lc "#98c127"      notitle,\
    ''  using 1:( $7/8  -0.06668314172560115) w l lw 3 lc "#98c127" notitle,\
    ''  using 1:( $9/8 - 0.07092646598213989) w l lw 3 lc "#98c127" notitle,\
    ''  using 1:( $10/2 -0.07299133014948526) w l lw 3 lc "#1a70bb" notitle,\
    ''  using 1:( $12/6 -0.07751587635660425) w l lw 3 lc "#98c127" notitle,\
    ''  using 1:( $17/1 -0.08382356967530796) w l lw 3 lc "#ea801c" notitle,\
    ''  using 1:( $18/1 -0.08704535880988144) w l lw 3 lc "#1a70bb" notitle,\
    ''  using 1:( $27/1 -0.09752809872044002) w l lw 3 lc "#ea801c" notitle,\
    ''  using 1:( $29/1 -0.10011037750249099) w l lw 3 lc "#1a70bb" notitle,\
    ''  using 1:( $36/1 -0.11144864768509424) w l lw 3 lc "#ea801c" notitle,\
    ''  using 1:( $44/1 -0.12391433639860834) w l lw 3 lc "#ea801c" notitle

unset xrange
set xrange [350:600]
unset ylabel
unset xlabel
set xlabel "|2>" offset 0,-2 font "CMU Serif,20"

set size 0.45,1
set origin 0.65,0.0

plot \
    NNdir . "B100.dat" \
        using 1:( $62/8 - 0.058584980569793646)    w l lw 3 lc "#1a70bb" notitle ,\
    ''  using 1:( $65*2 - 0.061184980569793646)     w l lw 3 lc "#98c127"      notitle,\
    ''  using 1:( $67*2  -0.06668314172560115) w l lw 3 lc "#98c127" notitle,\
    ''  using 1:( $69*2 - 0.07092646598213989) w l lw 3 lc "#98c127" notitle,\
    ''  using 1:( $70/8 -0.07299133014948526) w l lw 3 lc "#1a70bb" notitle,\
    ''  using 1:( $72*2 -0.07751587635660425) w l lw 3 lc "#98c127" notitle,\
    ''  using 1:( $137/8 -0.08382356967530796) w l lw 3 lc "#ea801c" notitle,\
    ''  using 1:( $78/8 -0.08704535880988144) w l lw 3 lc "#1a70bb" notitle,\
    ''  using 1:( $147/8 -0.09752809872044002) w l lw 3 lc "#ea801c" notitle,\
    ''  using 1:( $89/8 -0.10011037750249099) w l lw 3 lc "#1a70bb" notitle,\
    ''  using 1:( $156/5 -0.11144864768509424) w l lw 3 lc "#ea801c" notitle,\
    ''  using 1:( $167/5 -0.12391433639860834) w l lw 3 lc "#ea801c" notitle




unset multiplot
