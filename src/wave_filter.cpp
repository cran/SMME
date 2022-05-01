// #include <string>
//
// void qmf(int L, double* g, double* h){
//   for(int i = 0; i < L; i++){
//     h[i] = pow(-1, i) * g[L - 1 - i]; // ^(0:(L - 1)) * rev(g)
//     }
// }
// int get_L(std::string wf){
// int Lwave;
// if(wf == "haar"){Lwave = 2;}
// if(wf == "d4"){Lwave = 4;}
// if(wf == "??"){Lwave = 4;}
// if(wf == "mb4"){Lwave = 4;}
// if(wf == "fk4"){Lwave = 4;}
// if(wf == "d6"){Lwave = 6;}
// if(wf == "fk6"){Lwave= 6;}
// if(wf == "d8"){Lwave =8;}
// if(wf == "fk8"){Lwave =8;}
// if(wf == "la8"){Lwave =8;}
// if(wf == "mb8"){Lwave =8;}
// if(wf == "bl14"){Lwave =14;}
// if(wf == "fk14"){Lwave =14;}
// if(wf == "d16"){Lwave =16;}
// if(wf == "la16"){Lwave =16;}
// if(wf == "mb16"){Lwave =16;}
// if(wf == "la20"){Lwave =20;}
// if(wf == "bl20"){Lwave =20;}
// if(wf == "fk22"){Lwave =22;}
// if(wf == "mb24"){Lwave =24;}
// return Lwave;
// }
//
// void wave_filter(std::string filt,  double * g, double * h, int L){
//    if(filt == "haar"){
//       //int L = 2;
//       double tmp[2] = {0.7071067811865475, 0.7071067811865475};
//       std::copy(tmp, tmp + L , g);
//       qmf(L, g, h);
//       }
//     if(filt == "d4"){
//       //int L = 4;
//       double tmp[4] = {0.4829629131445341, 0.8365163037378077, 0.2241438680420134,
// 	                     -0.1294095225512603};
//       std::copy(tmp, tmp + L , g);
//       qmf(L, g, h);
//       }
//     if(filt == "e"){
//       //int L = 4;
//       double tmp[4] = {-0.125,  0.375,  0.375, -0.125};
//       std::copy(tmp, tmp + L , g);
//       double tmp2[4] = {-0.125,  0.375, -0.375,  0.125};
//       std::copy(tmp2, tmp2 + L , h);
//     }
//     if(filt == "mb4"){
//       //int L = 4;
//        double tmp[4] = {4.801755e-01, 8.372545e-01, 2.269312e-01, -1.301477e-01};
//        std::copy(tmp, tmp +  L, g);
//        qmf(L, g, h);
//        }
//     if(filt == "fk4"){
//       //int L = 4;
//       double tmp[4] = {.6539275555697651, .7532724928394872, 0.05317922877905981,
//           -0.0461657148152177};
//       std::copy(tmp, tmp + L, g);
//       qmf(L, g, h);
//       }
//     if(filt == "d6"){
//       //int L = 6;
//       double tmp[6] = {0.3326705529500827, 0.8068915093110928, 0.4598775021184915,
//           -0.1350110200102546, -0.0854412738820267, 0.0352262918857096};
//       std::copy(tmp, tmp + L , g);
//       qmf(L, g, h);
//       }
//     if(filt == "fk6"){
//       int L=6;
//        double tmp[6] = {.4279150324223103, .8129196431369074, .3563695110701871,
//           -.1464386812725773, -.7717775740697006e-1, .4062581442323794e-1};
//        std::copy(tmp, tmp + L, g);
//        qmf(L, g, h);
//        }
//     if(filt == "d8"){
//       //int L = 8;
//       double tmp[8] = {0.2303778133074431, 0.7148465705484058, 0.6308807679358788,
//           -0.0279837694166834, -0.1870348117179132, 0.0308413818353661,
//           0.0328830116666778, -0.0105974017850021};
//       std::copy(tmp, tmp + L , g);
//       qmf(L, g, h);
//       }
//     if(filt == "fk8"){
//       //int L = 8;
//       double tmp[8] = {.3492381118637999, .7826836203840648, .4752651350794712,
//           -.9968332845057319e-1, -.1599780974340301, .4310666810651625e-1,
//           .4258163167758178e-1, -.1900017885373592e-1};
//       std::copy(tmp, tmp + L , g);
//       qmf(L, g, h);
//       }
//     if(filt == "la8"){
//       //int L = 8;
//       double tmp[8] = {-0.07576571478935668, -0.02963552764596039, 0.49761866763256290,
// 	  0.80373875180538600, 0.29785779560560505, -0.09921954357695636,
// 	  -0.01260396726226383, 0.03222310060407815};
//       std::copy(tmp, tmp + L , g);
//       qmf(L, g, h);
//       }
//     if(filt == "mb8"){
//       //int L = 8;
//       double tmp[8] = {0.064363450,  0.007106015, -0.110867300,  0.294785500,  0.735133100,
//         0.572577100,  0.018477510, -0.167361900};
//       std::copy(tmp, tmp + L , g);
//       qmf(L, g, h);
//       }
//     if(filt == "bl14"){
//       //int L = 14;
//       double tmp[14] = { 0.0120154192834842, 0.0172133762994439, -0.0649080035533744,
// 	  -0.0641312898189170, 0.3602184608985549, 0.7819215932965554,
// 	   0.4836109156937821, -0.0568044768822707, -0.1010109208664125,
// 	   0.0447423494687405, 0.0204642075778225, -0.0181266051311065,
// 	  -0.0032832978473081, 0.0022918339541009};
//       std::copy(tmp, tmp + L , g);
//       qmf(L, g, h);
//       }
//     if(filt == "fk14"){
//       //int L = 14;
//       double tmp[14] = {.2603717692913964, .6868914772395985, .6115546539595115,
//           .5142165414211914e-1, -.2456139281621916, -.4857533908585527e-1,
//           .1242825609215128, .2222673962246313e-1, -.6399737303914167e-1,
//           -.5074372549972850e-2, .2977971159037902e-1, -.3297479152708717e-2,
//           -.9270613374448239e-2, .3514100970435962e-2};
//        std::copy(tmp, tmp + L , g);
//        qmf(L, g, h);
//        }
//     if(filt == "d16"){
//         //int L = 16;
//         double tmp[16] = {0.0544158422431049, 0.3128715909143031, 0.6756307362972904,
// 	  0.5853546836541907, -0.0158291052563816, -0.2840155429615702,
// 	  0.0004724845739124, 0.1287474266204837, -0.0173693010018083,
// 	 -0.0440882539307952, 0.0139810279173995, 0.0087460940474061,
// 	 -0.0048703529934518, -0.0003917403733770, 0.0006754494064506,
// 	 -0.0001174767841248};
//         std::copy(tmp, tmp + L , g);
//         qmf(L, g, h);
//         }
//     if(filt == "la16"){
//         //int L = 16;
//         double tmp[16] = {-0.0033824159513594, -0.0005421323316355, 0.0316950878103452,
// 	   0.0076074873252848, -0.1432942383510542, -0.0612733590679088,
// 	   0.4813596512592012, 0.7771857516997478, 0.3644418948359564,
// 	  -0.0519458381078751, -0.0272190299168137, 0.0491371796734768,
// 	   0.0038087520140601, -0.0149522583367926, -0.0003029205145516,
// 	   0.0018899503329007};
//        std::copy(tmp, tmp + L , g);
//        qmf(L, g, h);
//        }
//       if(filt == "mb16"){
//         //int L = 16;
//         double tmp[16] = {0.005765899,  0.009620427, -0.049846980, -0.024838760,  0.054746280,
//          -0.019879860, -0.056566570,  0.234534200,  0.670164600,  0.634922800,  0.118872500, -0.227835900,
//          -0.057765700,  0.113611600,  0.021736770, -0.013027700};
//        std::copy(tmp, tmp + L , g);
//        qmf(L, g, h);
//        }
//     if(filt == "la20"){
//         //int L = 20;
//         double tmp[20] = {0.0007701598091030, 0.0000956326707837, -0.0086412992759401,
//         -0.0014653825833465, 0.0459272392237649, 0.0116098939129724,
// 	      -0.1594942788575307, -0.0708805358108615, 0.4716906668426588,
// 	      0.7695100370143388, 0.3838267612253823, -0.0355367403054689,
//       	-0.0319900568281631, 0.0499949720791560, 0.0057649120455518,
//       	-0.0203549398039460, -0.0008043589345370, 0.0045931735836703,
//      	  0.0000570360843390, -0.0004593294205481};
//         std::copy(tmp, tmp + L , g);
//        qmf(L, g, h);
//        }
//     if(filt == "bl20"){
//         //int L = 20;
//         double tmp[20] = {0.0008625782242896, 0.0007154205305517, -0.0070567640909701,
//  	      0.0005956827305406, 0.0496861265075979, 0.0262403647054251,
//       	-0.1215521061578162, -0.0150192395413644, 0.5137098728334054,
// 	      0.7669548365010849, 0.3402160135110789, -0.0878787107378667,
//         -0.0670899071680668, 0.0338423550064691, -0.0008687519578684,
//         -0.0230054612862905, -0.0011404297773324, 0.0050716491945793,
//         0.0003401492622332, -0.0004101159165852};
//         std::copy(tmp, tmp + L , g);
//         qmf(L, g, h);
//         }
//     if(filt == "fk22"){
//            //int L = 22;
//        double tmp[22] = {.1938961077599566, .5894521909294277, .6700849629420265,
//           .2156298491347700, -.2280288557715772, -.1644657152688429,
//           .1115491437220700, .1101552649340661, -.6608451679377920e-1,
//           -.7184168192312605e-1, .4354236762555708e-1, .4477521218440976e-1,
//           -.2974288074927414e-1, -.2597087308902119e-1, .2028448606667798e-1,
//           .1296424941108978e-1, -.1288599056244363e-1, -.4838432636440189e-2,
//           .7173803165271690e-2, .3612855622194901e-3, -.2676991638581043e-2,
//           .8805773686384639e-3};
//        std::copy(tmp, tmp + L , g);
//        qmf(L, g, h);
//        }
//     if(filt == "mb24"){
//         //int L = 24;
//         double tmp[24] = {5.361301e-05,  1.193006e-03, -2.062335e-03, -1.644859e-02,
//           1.027154e-02,  1.479342e-01, -4.575448e-03, -3.622424e-01, -3.091111e-01,
//             2.556731e-01,  6.176385e-01, 4.581101e-01,  1.949147e-01,  1.243531e-01,
//               1.689456e-01,  1.019512e-01, -6.559513e-03, -2.658282e-03,  4.199576e-02,
//                -1.482995e-03, -4.879053e-03,  7.456041e-04,  4.745736e-04, -2.132706e-05};
//         std::copy(tmp, tmp + L , g);
//         qmf(L, g, h);
//         }
// }
//
//
// /*   select.bs3.1 <- function() {
//     L <- 4
//     g <- {0.1767767, 0.5303301, 0.5303301, 0.1767767)
//     h <- qmf(g)
//     gd <- {0.3535534, 1.06066, -1.06066, -0.3535534)
//     hd <- qmf(g)
//     return(list(length = L, hpf = h, lpf = g, dhpf = hd, dlpf = gd))
//   } */
//
//
//
