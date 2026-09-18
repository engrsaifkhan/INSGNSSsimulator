/*------------------------------------------------------------------------------
* pntpos.c : standard positioning
*
*          Copyright (C) 2007-2020 by T.TAKASU, All rights reserved.
*
* version : $Revision:$ $Date:$
* history : 2010/07/28 1.0  moved from rtkcmn.c
*                           changed api:
*                               pntpos()
*                           deleted api:
*                               pntvel()
*           2011/01/12 1.1  add option to include unhealthy satellite
*                           reject duplicated observation data
*                           changed api: ionocorr()
*           2011/11/08 1.2  enable snr mask for single-mode (rtklib_2.4.1_p3)
*           2012/12/25 1.3  add variable snr mask
*           2014/05/26 1.4  support galileo and beidou
*           2015/03/19 1.5  fix bug on ionosphere correction for GLO and BDS
*           2018/10/10 1.6  support api change of satexclude()
*           2020/11/30 1.7  support NavIC/IRNSS in pntpos()
*                           no support IONOOPT_LEX option in ioncorr()
*                           improve handling of TGD correction for each system
*                           use E1-E5b for Galileo dual-freq iono-correction
*                           use API sat2freq() to get carrier frequency
*                           add output of velocity estimation error in estvel()
*
*           branch for Pocket SDR
*           2026/06/01 0.1  add receiver clock rate in solution.
*-----------------------------------------------------------------------------*/
#include "rtklib.h"
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* constants/macros ----------------------------------------------------------*/

#define SQR(x)      ((x)*(x))

#if 0 /* enable GPS-QZS time offset estimation */
#define NX          (4+5)       /* # of estimated parameters */
#else
#define NX          (4+4)       /* # of estimated parameters */
#endif
#define MAXITR      10          /* max number of iteration for point pos */
#define ERR_ION     5.0         /* ionospheric delay Std (m) */
#define ERR_TROP    3.0         /* tropspheric delay Std (m) */
#define ERR_SAAS    0.3         /* Saastamoinen model error Std (m) */
#define ERR_BRDCI   0.5         /* broadcast ionosphere model error factor */
#define ERR_CBIAS   0.3         /* code bias error Std (m) */
#define REL_HUMI    0.7         /* relative humidity for Saastamoinen model */
#define MIN_EL      (5.0*D2R)   /* min elevation for measurement error (rad) */


/* Stability guards for accepted SPP fixes. These do not smooth sol->rr.
 * They only request one additional RAIM-FDE pass if the accepted solution still
 * contains a large post-fit pseudorange residual. The TCA export correction
 * formula is not changed.
 */
#ifndef PNTPOS_RAIM_RMS_GATE
#define PNTPOS_RAIM_RMS_GATE      12.0   /* m; RMS residual trigger */
#endif
#ifndef PNTPOS_RAIM_MAX_RES_GATE
#define PNTPOS_RAIM_MAX_RES_GATE  25.0   /* m; max residual trigger */
#endif
#ifndef PNTPOS_RAIM_MIN_NSAT
#define PNTPOS_RAIM_MIN_NSAT       6     /* need redundancy for FDE */
#endif


/* TCA corrected-measurement serial packet export ----------------------------
 * PocketSDR/PC side exports a text block for Teensy TCA:
 *
 *   TCA,<num_sats>,<obssec>
 *   <pr_corr>,<prrate_corr>,<x>,<y>,<z>,<vx>,<vy>,<vz>,<clk_rate>,<el>,<cn0>,<signalName>
 *   ...
 *   END
 *
 * Conventions:
 *   pr_corr     = prange() + c*sat_clk - iono - tropo - ISB
 *                 receiver clock bias is NOT removed
 *                 available inter-system biases are compensated to GPS time
 *   prrate_corr = Doppler range-rate + c*sat_clock_drift
 *                 receiver clock drift is NOT removed
 *   sat_clk_rate is dts[i*2+1] in s/s
 *
 *---------------------------------------------------------------------------*/
#define TCA_CSV_BUFF_SIZE 65536
#define GNSS_SIGNAL_LEN   32

/* Set these here if you want fixed TCA-side rejection independent of opt->snrmask.
 * The normal RTKLIB SNR mask is still applied by snrmask().
 */
#ifndef TCA_CN0_MIN
#define TCA_CN0_MIN 0.0       /* dB-Hz; 0.0 = use only opt->snrmask */
#endif

static char tca_packet_buff[TCA_CSV_BUFF_SIZE] = "";
static char tca_rows_buff  [TCA_CSV_BUFF_SIZE] = "";
static int  tca_packet_len = 0;
static int  tca_rows_len   = 0;

/* Reference receiver ECEF position for TCA corrections when PVT is not locked.
 * This is updated only after a valid pntpos() solution.
 */
static double tca_ref_rr[3] = {0.0, 0.0, 0.0};
static int    tca_has_ref_rr = 0;

/* Last valid receiver clock/inter-system offsets from pntpos().
 * dtr[0]: receiver clock bias relative to GPS time (kept in TCA data)
 * dtr[1]: GLONASS-GPS offset
 * dtr[2]: Galileo-GPS offset
 * dtr[3]: BeiDou-GPS offset
 * dtr[4]: NavIC-GPS offset
 * TCA export subtracts only dtr[1..4] so mixed-constellation
 * pseudoranges are aligned to the same GPS-referenced receiver clock.
 */
static double tca_ref_dtr[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
static int    tca_has_ref_dtr = 0;

extern const char *pntpos_get_tca_csv(void)
{
    return tca_packet_buff;
}

static void tca_buff_reset(char *buff, int *len)
{
    buff[0] = '\0';
    *len = 0;
}

static void tca_export_reset(void)
{
    tca_buff_reset(tca_packet_buff, &tca_packet_len);
    tca_buff_reset(tca_rows_buff,   &tca_rows_len);
}

static void tca_buff_append(char *buff, int *len, const char *fmt, ...)
{
    va_list ap;
    int n;

    if (*len >= TCA_CSV_BUFF_SIZE - 1) return;

    va_start(ap, fmt);
    n = vsnprintf(buff + *len, TCA_CSV_BUFF_SIZE - *len, fmt, ap);
    va_end(ap);

    if (n <= 0) return;

    if (*len + n >= TCA_CSV_BUFF_SIZE) {
        *len = TCA_CSV_BUFF_SIZE - 1;
        buff[*len] = '\0';
    }
    else {
        *len += n;
    }
}

static const char *tca_signal_name(int sys, uint8_t code)
{
    const char *obs = code2obs(code);

    if (!obs || !*obs) return "SIG_UNKNOWN";

    /* GPS */
    if (sys & SYS_GPS) {
        if (!strcmp(obs, "1C")) return "SIG_GPS_L1_CA";
    }
    /* Galileo */
    if (sys & SYS_GAL) {
        if (!strcmp(obs, "1B")) return "SIG_GALILEO_E1B";
        if (!strcmp(obs, "1C")) return "SIG_GALILEO_E1C";
        if (!strcmp(obs, "5I")) return "SIG_GALILEO_E5A_I";
        if (!strcmp(obs, "7I")) return "SIG_GALILEO_E5B_I";
    }
    /* BeiDou / BDS */
    if (sys & SYS_CMP) {
        if (!strcmp(obs, "2I") || !strcmp(obs, "1I")) return "SIG_BEIDOU_B1I";
        if (!strcmp(obs, "1D")) return "SIG_BEIDOU_B1CD";
        if (!strcmp(obs, "5D")) return "SIG_BEIDOU_B2A_D";
        if (!strcmp(obs, "7D") || !strcmp(obs, "7P")) return "SIG_BEIDOU_B2B";
        if (!strcmp(obs, "7I")) return "SIG_BEIDOU_B2I";
        if (!strcmp(obs, "6I")) return "SIG_BEIDOU_B3I";
    }
    /* GLONASS */
    if (sys & SYS_GLO) {
        if (!strcmp(obs, "1C")) return "SIG_GLONASS_G1_CA";
    }
    return "SIG_UNKNOWN";
}

/* inter-system bias correction for TCA pseudorange export --------------------
 * Returns estimated constellation time-offset bias in meters. The common
 * receiver clock bias dtr[0] is intentionally not removed because the Teensy
 * TCA/INS filter should still estimate receiver clock bias/drift.
 */
static double tca_inter_system_bias_m(int sys)
{
    if (!tca_has_ref_dtr) return 0.0;

    if (sys & SYS_GLO) return CLIGHT * tca_ref_dtr[1];
    if (sys & SYS_GAL) return CLIGHT * tca_ref_dtr[2];
    if (sys & SYS_CMP) return CLIGHT * tca_ref_dtr[3];
    if (sys & SYS_IRN) return CLIGHT * tca_ref_dtr[4];

    return 0.0; /* GPS/QZSS/SBAS reference: no additional ISB removed */
}

/* pseudorange measurement error variance ------------------------------------*/
static double varerr(const prcopt_t *opt, double el, int sys)
{
    double fact,varr;
    fact=sys==SYS_GLO?EFACT_GLO:(sys==SYS_SBS?EFACT_SBS:EFACT_GPS);
    if (el<MIN_EL) el=MIN_EL;
    varr=SQR(opt->err[0])*(SQR(opt->err[1])+SQR(opt->err[2])/sin(el));
    if (opt->ionoopt==IONOOPT_IFLC) varr*=SQR(3.0); /* iono-free */
    return SQR(fact)*varr;
}
/* get group delay parameter (m) ---------------------------------------------*/
static double gettgd(int sat, const nav_t *nav, int type)
{
    int i,sys=satsys(sat,NULL);
    
    if (sys==SYS_GLO) {
        for (i=0;i<nav->ng;i++) {
            if (nav->geph[i].sat==sat) break;
        }
        return (i>=nav->ng)?0.0:-nav->geph[i].dtaun*CLIGHT;
    }
    else {
        for (i=0;i<nav->n;i++) {
            if (nav->eph[i].sat==sat) break;
        }
        return (i>=nav->n)?0.0:nav->eph[i].tgd[type]*CLIGHT;
    }
}
/* test SNR mask -------------------------------------------------------------*/
static int snrmask(const obsd_t *obs, const double *azel, const prcopt_t *opt)
{
    if (testsnr(0,0,azel[1],obs->SNR[0]*SNR_UNIT,&opt->snrmask)) {
        return 0;
    }
    if (opt->ionoopt==IONOOPT_IFLC) {
        if (testsnr(0,1,azel[1],obs->SNR[1]*SNR_UNIT,&opt->snrmask)) return 0;
    }
    return 1;
}
/* psendorange with code bias correction -------------------------------------*/
static double prange(const obsd_t *obs, const nav_t *nav, const prcopt_t *opt,
                     double *var)
{
    double P1,P2,gamma,b1,b2;
    int sat,sys;
    
    sat=obs->sat;
    sys=satsys(sat,NULL);
    P1=obs->P[0];
    P2=obs->P[1];
    *var=0.0;
    
    if (P1==0.0||(opt->ionoopt==IONOOPT_IFLC&&P2==0.0)) return 0.0;
    
    /* P1-C1,P2-C2 DCB correction */
    if (sys==SYS_GPS||sys==SYS_GLO) {
        if (obs->code[0]==CODE_L1C) P1+=nav->cbias[sat-1][1]; /* C1->P1 */
        if (obs->code[1]==CODE_L2C) P2+=nav->cbias[sat-1][2]; /* C2->P2 */
    }
    if (opt->ionoopt==IONOOPT_IFLC) { /* dual-frequency */
        
        if (sys==SYS_GPS||sys==SYS_QZS) { /* L1-L2,G1-G2 */
            gamma=SQR(FREQ1/FREQ2);
            return (P2-gamma*P1)/(1.0-gamma);
        }
        else if (sys==SYS_GLO) { /* G1-G2 */
            gamma=SQR(FREQ1_GLO/FREQ2_GLO);
            return (P2-gamma*P1)/(1.0-gamma);
        }
        else if (sys==SYS_GAL) { /* E1-E5b (Galileo): use explicit E5b frequency */
            /* Use E1-E5b iono-free combination (E5b = FREQ7) */
            gamma = SQR(FREQ1 / FREQ7);
            /* Use available TGD entries: tgd[0]=BGD_E1E5a, tgd[1]=BGD_E1E5b for GAL
               nav->eph[].tgd indices are system-dependent; keep existing TGD
               handling but ensure E5b is used for the iono-free frequency ratio. */
            if      (obs->code[0]==CODE_L2I) b1=gettgd(sat,nav,0); /* keep legacy mapping */
            else if (obs->code[0]==CODE_L1P) b1=gettgd(sat,nav,2);
            else b1=gettgd(sat,nav,2)+gettgd(sat,nav,4);
            b2=gettgd(sat,nav,1); /* E1-E5b TGD (m) */
            return ((P2-gamma*P1)-(b2-gamma*b1))/(1.0-gamma);
        }
        else if (sys==SYS_CMP) { /* B1-B2 */
            gamma=SQR(((obs->code[0]==CODE_L2I)?FREQ1_CMP:FREQ1)/FREQ2_CMP);
            if      (obs->code[0]==CODE_L2I) b1=gettgd(sat,nav,0); /* TGD_B1I */
            else if (obs->code[0]==CODE_L1P) b1=gettgd(sat,nav,2); /* TGD_B1Cp */
            else b1=gettgd(sat,nav,2)+gettgd(sat,nav,4); /* TGD_B1Cp+ISC_B1Cd */
            b2=gettgd(sat,nav,1); /* TGD_B2I/B2bI (m) */
            return ((P2-gamma*P1)-(b2-gamma*b1))/(1.0-gamma);
        }
        else if (sys==SYS_IRN) { /* L5-S */
            gamma=SQR(FREQ5/FREQ9);
            return (P2-gamma*P1)/(1.0-gamma);
        }
    }
    else { /* single-freq (L1/E1/B1) */
        *var=SQR(ERR_CBIAS);
        
        if (sys==SYS_GPS||sys==SYS_QZS) { /* L1 */
            b1=gettgd(sat,nav,0); /* TGD (m) */
            return P1-b1;
        }
        else if (sys==SYS_GLO) { /* G1 */
            gamma=SQR(FREQ1_GLO/FREQ2_GLO);
            b1=gettgd(sat,nav,0); /* -dtaun (m) */
            return P1-b1/(gamma-1.0);
        }
        else if (sys==SYS_GAL) { /* E1, treat like BeiDou B1I */
            if      (obs->code[0]==CODE_L2I) b1=gettgd(sat,nav,0); /* TGD_B1I */
            else if (obs->code[0]==CODE_L1P) b1=gettgd(sat,nav,2); /* TGD_B1Cp */
            else b1=gettgd(sat,nav,2)+gettgd(sat,nav,4); /* TGD_B1Cp+ISC_B1Cd */
            return P1-b1;
        }
        else if (sys==SYS_CMP) { /* B1I/B1Cp/B1Cd */
            if      (obs->code[0]==CODE_L2I) b1=gettgd(sat,nav,0); /* TGD_B1I */
            else if (obs->code[0]==CODE_L1P) b1=gettgd(sat,nav,2); /* TGD_B1Cp */
            else b1=gettgd(sat,nav,2)+gettgd(sat,nav,4); /* TGD_B1Cp+ISC_B1Cd */
            return P1-b1;
        }
        else if (sys==SYS_IRN) { /* L5 */
            gamma=SQR(FREQ9/FREQ5);
            b1=gettgd(sat,nav,0); /* TGD (m) */
            return P1-gamma*b1;
        }
    }
    return P1;
}
/* ionospheric correction ------------------------------------------------------
* compute ionospheric correction
* args   : gtime_t time     I   time
*          nav_t  *nav      I   navigation data
*          int    sat       I   satellite number
*          double *pos      I   receiver position {lat,lon,h} (rad|m)
*          double *azel     I   azimuth/elevation angle {az,el} (rad)
*          int    ionoopt   I   ionospheric correction option (IONOOPT_???)
*          double *ion      O   ionospheric delay (L1) (m)
*          double *var      O   ionospheric delay (L1) variance (m^2)
* return : status(1:ok,0:error)
*-----------------------------------------------------------------------------*/
extern int ionocorr(gtime_t time, const nav_t *nav, int sat, const double *pos,
                    const double *azel, int ionoopt, double *ion, double *var)
{
    trace(4,"ionocorr: time=%s opt=%d sat=%2d pos=%.3f %.3f azel=%.3f %.3f\n",
          time_str(time,3),ionoopt,sat,pos[0]*R2D,pos[1]*R2D,azel[0]*R2D,
          azel[1]*R2D);
    
    /* GPS broadcast ionosphere model */
    if (ionoopt==IONOOPT_BRDC) {
        *ion=ionmodel(time,nav->ion_gps,pos,azel);
        *var=SQR(*ion*ERR_BRDCI);
        return 1;
    }
    /* SBAS ionosphere model */
    if (ionoopt==IONOOPT_SBAS) {
        return sbsioncorr(time,nav,pos,azel,ion,var);
    }
    /* IONEX TEC model */
    if (ionoopt==IONOOPT_TEC) {
        return iontec(time,nav,pos,azel,1,ion,var);
    }
    /* QZSS broadcast ionosphere model */
    if (ionoopt==IONOOPT_QZS&&norm(nav->ion_qzs,8)>0.0) {
        *ion=ionmodel(time,nav->ion_qzs,pos,azel);
        *var=SQR(*ion*ERR_BRDCI);
        return 1;
    }
    *ion=0.0;
    *var=ionoopt==IONOOPT_OFF?SQR(ERR_ION):0.0;
    return 1;
}
/* tropospheric correction -----------------------------------------------------
* compute tropospheric correction
* args   : gtime_t time     I   time
*          nav_t  *nav      I   navigation data
*          double *pos      I   receiver position {lat,lon,h} (rad|m)
*          double *azel     I   azimuth/elevation angle {az,el} (rad)
*          int    tropopt   I   tropospheric correction option (TROPOPT_???)
*          double *trp      O   tropospheric delay (m)
*          double *var      O   tropospheric delay variance (m^2)
* return : status(1:ok,0:error)
*-----------------------------------------------------------------------------*/
extern int tropcorr(gtime_t time, const nav_t *nav, const double *pos,
                    const double *azel, int tropopt, double *trp, double *var)
{
    trace(4,"tropcorr: time=%s opt=%d pos=%.3f %.3f azel=%.3f %.3f\n",
          time_str(time,3),tropopt,pos[0]*R2D,pos[1]*R2D,azel[0]*R2D,
          azel[1]*R2D);
    
    /* Saastamoinen model */
    if (tropopt==TROPOPT_SAAS||tropopt==TROPOPT_EST||tropopt==TROPOPT_ESTG) {
        *trp=tropmodel(time,pos,azel,REL_HUMI);
        *var=SQR(ERR_SAAS/(sin(azel[1])+0.1));
        return 1;
    }
    /* SBAS (MOPS) troposphere model */
    if (tropopt==TROPOPT_SBAS) {
        *trp=sbstropcorr(time,pos,azel,var);
        return 1;
    }
    /* no correction */
    *trp=0.0;
    *var=tropopt==TROPOPT_OFF?SQR(ERR_TROP):0.0;
    return 1;
}
/* pseudorange residuals -----------------------------------------------------*/
static int rescode(int iter, const obsd_t *obs, int n, const double *rs,
                   const double *dts, const double *vare, const int *svh,
                   const nav_t *nav, const double *x, const prcopt_t *opt,
                   double *v, double *H, double *var, double *azel, int *vsat,
                   double *resp, int *ns)
{
    gtime_t time;
#if 0
    double r,freq,dion,dtrp,vmeas,vion,vtrp,rr[3],pos[3],dtr,e[3],P;
#else
    double r,freq,dion,dtrp=0.0,vmeas,vion,vtrp,rr[3],pos[3],dtr,e[3],P;
    double dant[NFREQ]={0.0};
#endif
    int i,j,nv=0,sat,sys,mask[NX-3]={0};
    
    trace(3,"resprng : n=%d\n",n);
    
    for (i=0;i<3;i++) rr[i]=x[i];
    dtr=x[3];
    
    ecef2pos(rr,pos);
    
    for (i=*ns=0;i<n&&i<MAXOBS;i++) {
        vsat[i]=0; azel[i*2]=azel[1+i*2]=resp[i]=0.0;
        time=obs[i].time;
        sat=obs[i].sat;
        if (!(sys=satsys(sat,NULL))) continue;
        
        /* reject duplicated observation data */
        if (i<n-1&&i<MAXOBS-1&&sat==obs[i+1].sat) {
            trace(2,"duplicated obs data %s sat=%d\n",time_str(time,3),sat);
            i++;
            continue;
        }
        /* excluded satellite? */
        if (satexclude(sat,vare[i],svh[i],opt)) continue;
        
        /* geometric distance */
        if ((r=geodist(rs+i*6,rr,e))<=0.0) continue;
        
        if (iter>0) {
            /* test elevation mask */
            if (satazel(pos,e,azel+i*2)<opt->elmin) continue;
            
            /* test SNR mask */
            if (!snrmask(obs+i,azel+i*2,opt)) continue;
            
            /* ionospheric correction */
            if (!ionocorr(time,nav,sat,pos,azel+i*2,opt->ionoopt,&dion,&vion)) {
                continue;
            }
            if ((freq=sat2freq(sat,obs[i].code[0],nav))==0.0) continue;
            dion*=SQR(FREQ1/freq);
            vion*=SQR(FREQ1/freq);
            
            /* tropospheric correction */
            if (!tropcorr(time,nav,pos,azel+i*2,opt->tropopt,&dtrp,&vtrp)) {
                continue;
            }
        }
        /* pseudorange with code bias and antenna correction */
        if ((P=prange(obs+i,nav,opt,&vmeas))==0.0) continue;
        if (opt->pcvr) {
            antmodel(opt->pcvr,opt->antdel[0],azel+i*2,opt->posopt[1],dant);
            j=code2idx(sys,obs[i].code[0]);
            if (j >= 0 && j < NFREQ) {
                P-=dant[j];
            }
        }
        
        /* pseudorange residual */
        v[nv]=P-(r+dtr-CLIGHT*dts[i*2]+dion+dtrp);
        
        /* design matrix */
        for (j=0;j<NX;j++) {
            H[j+nv*NX]=j<3?-e[j]:(j==3?1.0:0.0);
        }
        /* time system offset and receiver bias correction */
        if      (sys==SYS_GLO) {v[nv]-=x[4]; H[4+nv*NX]=1.0; mask[1]=1;}
        else if (sys==SYS_GAL) {v[nv]-=x[5]; H[5+nv*NX]=1.0; mask[2]=1;}
        else if (sys==SYS_CMP) {v[nv]-=x[6]; H[6+nv*NX]=1.0; mask[3]=1;}
        else if (sys==SYS_IRN) {v[nv]-=x[7]; H[7+nv*NX]=1.0; mask[4]=1;}
#if 0 /* enable QZS-GPS time offset estimation */
        else if (sys==SYS_QZS) {v[nv]-=x[8]; H[8+nv*NX]=1.0; mask[5]=1;}
#endif
        else mask[0]=1;
        
        vsat[i]=1; resp[i]=v[nv]; (*ns)++;
        
        /* variance of pseudorange error */
        var[nv++]=varerr(opt,azel[1+i*2],sys)+vare[i]+vmeas+vion+vtrp;
        
        trace(4,"sat=%2d azel=%5.1f %4.1f res=%7.3f sig=%5.3f\n",obs[i].sat,
              azel[i*2]*R2D,azel[1+i*2]*R2D,resp[i],sqrt(var[nv-1]));
    }
    /* constraint to avoid rank-deficient */
    for (i=0;i<NX-3;i++) {
        if (mask[i]) continue;
        v[nv]=0.0;
        for (j=0;j<NX;j++) H[j+nv*NX]=j==i+3?1.0:0.0;
        var[nv++]=0.01;
    }
    return nv;
}
/* validate solution ---------------------------------------------------------*/
static int valsol(const double *azel, const int *vsat, int n,
                  const prcopt_t *opt, const double *v, int nv, int nx,
                  char *msg)
{
    double azels[MAXOBS*2],dop[4],vv;
    int i,ns;
    
    trace(3,"valsol  : n=%d nv=%d\n",n,nv);
    
    /* Chi-square validation of residuals */
    vv=dot(v,v,nv);
    if (nv>nx&&vv>chisqr[nv-nx-1]) {
        sprintf(msg,"chi-square error nv=%d vv=%.1f cs=%.1f",nv,vv,chisqr[nv-nx-1]);
        return 0;
    }
    /* large GDOP check */
    for (i=ns=0;i<n;i++) {
        if (!vsat[i]) continue;
        azels[  ns*2]=azel[  i*2];
        azels[1+ns*2]=azel[1+i*2];
        ns++;
    }
    dops(ns,azels,opt->elmin,dop);
    if (dop[0]<=0.0||dop[0]>opt->maxgdop) {
        sprintf(msg,"gdop error nv=%d gdop=%.1f",nv,dop[0]);
        return 0;
    }
    return 1;
}
/* estimate receiver position ------------------------------------------------*/
static int estpos(const obsd_t *obs, int n, const double *rs, const double *dts,
                  const double *vare, const int *svh, const nav_t *nav,
                  const prcopt_t *opt, sol_t *sol, double *azel, int *vsat,
                  double *resp, char *msg)
{
    double x[NX]={0},dx[NX],Q[NX*NX],*v,*H,*var,sig;
    int i,j,k,info,stat,nv,ns;
    
    trace(3,"estpos  : n=%d\n",n);
    
    v=mat(n+4,1); H=mat(NX,n+4); var=mat(n+4,1);
    
    for (i=0;i<3;i++) x[i]=sol->rr[i];
    
    for (i=0;i<MAXITR;i++) {
        
        /* pseudorange residuals (m) */
        nv=rescode(i,obs,n,rs,dts,vare,svh,nav,x,opt,v,H,var,azel,vsat,resp,
                   &ns);
        
        if (nv<NX) {
            sprintf(msg,"lack of valid sats ns=%d",nv);
            break;
        }
        /* weighted by Std */
        for (j=0;j<nv;j++) {
            sig=sqrt(var[j]);
            v[j]/=sig;
            for (k=0;k<NX;k++) H[k+j*NX]/=sig;
        }
        /* least square estimation */
        if ((info=lsq(H,v,NX,nv,dx,Q))) {
            sprintf(msg,"lsq error info=%d",info);
            break;
        }
        for (j=0;j<NX;j++) {
            x[j]+=dx[j];
        }
        if (norm(dx,NX)<1E-4) {
            sol->type=0;
            sol->time=timeadd(obs[0].time,-x[3]/CLIGHT);
            sol->dtr[0]=x[3]/CLIGHT; /* receiver clock bias (s) */
            sol->dtr[1]=x[4]/CLIGHT; /* GLO-GPS time offset (s) */
            sol->dtr[2]=x[5]/CLIGHT; /* GAL-GPS time offset (s) */
            sol->dtr[3]=x[6]/CLIGHT; /* BDS-GPS time offset (s) */
            sol->dtr[4]=x[7]/CLIGHT; /* IRN-GPS time offset (s) */
            for (j=0;j<6;j++) sol->rr[j]=j<3?x[j]:0.0;
            for (j=0;j<3;j++) sol->qr[j]=(float)Q[j+j*NX];
            sol->qr[3]=(float)Q[1];    /* cov xy */
            sol->qr[4]=(float)Q[2+NX]; /* cov yz */
            sol->qr[5]=(float)Q[2];    /* cov zx */
            sol->ns=(uint8_t)ns;
            sol->age=sol->ratio=0.0;
            
            /* validate solution */
            if ((stat=valsol(azel,vsat,n,opt,v,nv,NX,msg))) {
                sol->stat=opt->sateph==EPHOPT_SBAS?SOLQ_SBAS:SOLQ_SINGLE;
            }
            free(v); free(H); free(var);
            return stat;
        }
    }
    if (i>=MAXITR) sprintf(msg,"iteration divergent i=%d",i);
    
    free(v); free(H); free(var);
    return 0;
}
/* RAIM FDE (failure detection and exclution) -------------------------------*/
static int raim_fde(const obsd_t *obs, int n, const double *rs,
                    const double *dts, const double *vare, const int *svh,
                    const nav_t *nav, const prcopt_t *opt, sol_t *sol,
                    double *azel, int *vsat, double *resp, char *msg)
{
    obsd_t *obs_e;
    sol_t sol_e={{0}};
    char tstr[32],name[16],msg_e[128];
    double *rs_e,*dts_e,*vare_e,*azel_e,*resp_e,rms_e,rms=100.0;
    int i,j,k,nvsat,stat=0,*svh_e,*vsat_e,sat=0;
    
    trace(3,"raim_fde: %s n=%2d\n",time_str(obs[0].time,0),n);
    
    if (!(obs_e=(obsd_t *)malloc(sizeof(obsd_t)*n))) return 0;
    rs_e = mat(6,n); dts_e = mat(2,n); vare_e=mat(1,n); azel_e=zeros(2,n);
    svh_e=imat(1,n); vsat_e=imat(1,n); resp_e=mat(1,n); 
    
    for (i=0;i<n;i++) {
        
        /* satellite exclution */
        for (j=k=0;j<n;j++) {
            if (j==i) continue;
            obs_e[k]=obs[j];
            matcpy(rs_e +6*k,rs +6*j,6,1);
            matcpy(dts_e+2*k,dts+2*j,2,1);
            vare_e[k]=vare[j];
            svh_e[k++]=svh[j];
        }
        /* estimate receiver position without a satellite */
        if (!estpos(obs_e,n-1,rs_e,dts_e,vare_e,svh_e,nav,opt,&sol_e,azel_e,
                    vsat_e,resp_e,msg_e)) {
            trace(3,"raim_fde: exsat=%2d (%s)\n",obs[i].sat,msg);
            continue;
        }
        for (j=nvsat=0,rms_e=0.0;j<n-1;j++) {
            if (!vsat_e[j]) continue;
            rms_e+=SQR(resp_e[j]);
            nvsat++;
        }
        if (nvsat<5) {
            trace(3,"raim_fde: exsat=%2d lack of satellites nvsat=%2d\n",
                  obs[i].sat,nvsat);
            continue;
        }
        rms_e=sqrt(rms_e/nvsat);
        
        trace(3,"raim_fde: exsat=%2d rms=%8.3f\n",obs[i].sat,rms_e);
        
        if (rms_e>rms) continue;
        
        /* save result */
        for (j=k=0;j<n;j++) {
            if (j==i) continue;
            matcpy(azel+2*j,azel_e+2*k,2,1);
            vsat[j]=vsat_e[k];
            resp[j]=resp_e[k++];
        }
        stat=1;
        *sol=sol_e;
        sat=obs[i].sat;
        rms=rms_e;
        vsat[i]=0;
        strcpy(msg,msg_e);
    }
    if (stat) {
        time2str(obs[0].time,tstr,2); satno2id(sat,name);
        trace(2,"%s: %s excluded by raim\n",tstr+11,name);
    }
    free(obs_e);
    free(rs_e ); free(dts_e ); free(vare_e); free(azel_e);
    free(svh_e); free(vsat_e); free(resp_e);

    return stat;
}
/* check if an accepted solution still needs an FDE pass ----------------------*/
static int need_residual_fde(const int *vsat, const double *resp, int n)
{
    double rms=0.0,maxr=0.0,r;
    int i,ns=0;

    for (i=0;i<n&&i<MAXOBS;i++) {
        if (!vsat[i]) continue;
        r=fabs(resp[i]);
        rms+=r*r;
        if (r>maxr) maxr=r;
        ns++;
    }
    if (ns<PNTPOS_RAIM_MIN_NSAT) return 0;
    rms=sqrt(rms/ns);

    return rms>PNTPOS_RAIM_RMS_GATE||maxr>PNTPOS_RAIM_MAX_RES_GATE;
}
/* range rate residuals ------------------------------------------------------*/
static int resdop(const obsd_t *obs, int n, const double *rs, const double *dts,
                  const nav_t *nav, const double *rr, const double *x,
                  const double *azel, const int *vsat, double err, double *v,
                  double *H)
{
    double freq,rate,pos[3],E[9],a[3],e[3],vs[3],cosel,sig;
    int i,j,nv=0;
    
    trace(3,"resdop  : n=%d\n",n);
    
    ecef2pos(rr,pos); xyz2enu(pos,E);
    
    for (i=0;i<n&&i<MAXOBS;i++) {
        
        freq=sat2freq(obs[i].sat,obs[i].code[0],nav);
        
        if (obs[i].D[0]==0.0||freq==0.0||!vsat[i]||norm(rs+3+i*6,3)<=0.0) {
            continue;
        }
        /* LOS (line-of-sight) vector in ECEF */
        cosel=cos(azel[1+i*2]);
        a[0]=sin(azel[i*2])*cosel;
        a[1]=cos(azel[i*2])*cosel;
        a[2]=sin(azel[1+i*2]);
        matmul("TN",3,1,3,1.0,E,a,0.0,e);
        
        /* satellite velocity relative to receiver in ECEF */
        for (j=0;j<3;j++) {
            vs[j]=rs[j+3+i*6]-x[j];
        }
        /* range rate with earth rotation correction */
        /* Use receiver ECEF position (`rr`) here — previous code used
         * `x[]` (velocity state) by mistake which can corrupt velocity
         * estimation and downstream TCA outputs. */
        rate = dot(vs,e,3) + OMGE/CLIGHT*(rs[4+i*6]*rr[0] + rs[1+i*6]*rr[1]
                         - rs[3+i*6]*rr[1] - rs[i*6]*rr[0]);
        
        /* Std of range rate error (m/s) */
        sig=(err<=0.0)?1.0:err*CLIGHT/freq;
        
        /* range rate residual (m/s) */
        v[nv]=(-obs[i].D[0]*CLIGHT/freq-(rate+x[3]-CLIGHT*dts[1+i*2]))/sig;
        
        /* design matrix */
        for (j=0;j<4;j++) {
            H[j+nv*4]=((j<3)?-e[j]:1.0)/sig;
        }
        nv++;
    }
    return nv;
}
/* estimate receiver velocity ------------------------------------------------*/
static void estvel(const obsd_t *obs, int n, const double *rs, const double *dts,
                   const nav_t *nav, const prcopt_t *opt, sol_t *sol,
                   const double *azel, const int *vsat)
{
    double x[4]={0},dx[4],Q[16],*v,*H;
    double err=opt->err[4]; /* Doppler error (Hz) */
    int i,j,nv;
    
    trace(3,"estvel  : n=%d\n",n);
    
    v=mat(n,1); H=mat(4,n);
    
    for (i=0;i<MAXITR;i++) {
        
        /* range rate residuals (m/s) */
        if ((nv=resdop(obs,n,rs,dts,nav,sol->rr,x,azel,vsat,err,v,H))<4) {
            break;
        }
        /* least square estimation */
        if (lsq(H,v,4,nv,dx,Q)) break;
        
        for (j=0;j<4;j++) x[j]+=dx[j];
        
        if (norm(dx,4)<1E-6) {
            matcpy(sol->rr+3,x,3,1);
            sol->dtr[5]=x[3]/CLIGHT;  /* receiver clock drift rate (s/s) */
            sol->qv[0]=(float)Q[0];  /* xx */
            sol->qv[1]=(float)Q[5];  /* yy */
            sol->qv[2]=(float)Q[10]; /* zz */
            sol->qv[3]=(float)Q[1];  /* xy */
            sol->qv[4]=(float)Q[6];  /* yz */
            sol->qv[5]=(float)Q[2];  /* zx */
            break;
        }
    }
    free(v); free(H);
}


/* export TCA-ready corrected measurements -----------------------------------
 * This function is intentionally independent of the final PVT status.
 *
 * require_vsat = 1: normal locked-PVT mode; only satellites accepted by
 *                   estpos()/RAIM are exported.
 * require_vsat = 0: partial-measurement mode; PVT lock is not required.
 *                   Valid pre-fit measurements are compensated and exported.
 *
 * The following corrections/checks are still applied in both modes:
 *   - satellite health/exclusion via satexclude()
 *   - valid pseudorange/Doppler/code checks
 *   - satellite position/clock availability
 *   - elevation mask opt->elmin
 *   - RTKLIB SNR mask via snrmask()
 *   - optional fixed CN0 threshold TCA_CN0_MIN
 *   - prange() TGD/BGD/DCB/code-bias correction
 *   - satellite clock bias correction
 *   - ionosphere and troposphere correction
 *   - estimated inter-system bias correction when available
 *   - satellite clock drift correction for pseudorange rate
 *---------------------------------------------------------------------------*/
static void export_tca_csv_from_pntpos(const obsd_t *obs, int n,
                                       const nav_t *nav,
                                       const prcopt_t *opt,
                                       const double *rr_ref,
                                       const double *rr_vel,
                                       const double *rs,
                                       const double *dts,
                                       const double *vare,
                                       const int *svh,
                                       const double *azel,
                                       const int *vsat,
                                       int require_vsat)
{
    double pos[3];
    int i,week,count=0;

    tca_export_reset();

    if (!obs || n<=0 || !nav || !opt || !rr_ref || !rs || !dts ||
        !vare || !svh) {
        return;
    }
    if (norm(rr_ref,3)<=0.0) return;

    ecef2pos(rr_ref,pos);

    for (i=0;i<n&&i<MAXOBS;i++) {
        double freq,vmeas,P_prange,dion=0.0,vion=0.0,dtrp=0.0,vtrp=0.0;
        double pr_corr,prrate_corr,sys_bias,dant[NFREQ]={0.0};
        double azel_i[2]={0.0,0.0},e[3],r,cn0;
        const char *sig_name;
        int sys,sat,idx;

        sat=obs[i].sat;
        if (!(sys=satsys(sat,NULL))) continue;

        /* In locked PVT mode, keep the same accepted-satellite filtering. */
        if (require_vsat && vsat && !vsat[i]) continue;

        /* Do not export unhealthy/excluded satellites. */
        if (satexclude(sat,vare[i],svh[i],opt)) continue;

        if (obs[i].P[0]<=0.0) continue;
        if (obs[i].D[0]==0.0) continue;
        if (obs[i].code[0]==0) continue;

        if (norm(rs+i*6,3)<=0.0) continue;
        if ((freq=sat2freq(sat,obs[i].code[0],nav))==0.0) continue;

        /* Geometry and az/el from reference receiver position. */
        if ((r=geodist(rs+i*6,rr_ref,e))<=0.0) continue;

        if (azel) {
            azel_i[0]=azel[i*2];
            azel_i[1]=azel[1+i*2];
            if (azel_i[1]<=0.0) {
                satazel(pos,e,azel_i);
            }
        }
        else {
            satazel(pos,e,azel_i);
        }

        /* Same elevation-mask idea as pntpos residual generation. */
        if (azel_i[1]<opt->elmin) continue;

        /* Same RTKLIB SNR mask logic used by pntpos(). */
        if (!snrmask(obs+i,azel_i,opt)) continue;

        cn0=obs[i].SNR[0]*SNR_UNIT;
        if (cn0<TCA_CN0_MIN) continue;

        /* prange() applies TGD/BGD/DCB/code-bias correction. */
        if ((P_prange=prange(obs+i,nav,opt,&vmeas))==0.0) continue;

        /* receiver antenna phase center correction */
        antmodel(opt->pcvr,opt->antdel[0],azel_i,opt->posopt[1],dant);
        idx = code2idx(sys, obs[i].code[0]);

        /* Ionospheric correction. */
        if (!ionocorr(obs[i].time,nav,sat,pos,azel_i,opt->ionoopt,
                      &dion,&vion)) {
            continue;
        }
        dion*=SQR(FREQ1/freq);

        /* Tropospheric correction. */
        if (!tropcorr(obs[i].time,nav,pos,azel_i,opt->tropopt,
                      &dtrp,&vtrp)) {
            continue;
        }

        /* TCA exported measurements without Sagnac removal.
         * prange() has already applied TGD/BGD/DCB/code-bias correction.
         * Receiver antenna phase center correction is applied here.
         * Remove only the estimated inter-system bias (GLO/GAL/BDS/IRN)
         * so all exported pseudoranges use the GPS-referenced receiver clock.
         * The common receiver clock bias remains in pr_corr and must still be
         * estimated by the Teensy TCA/INS filter.
         */
        sys_bias = tca_inter_system_bias_m(sys);
        pr_corr = P_prange + CLIGHT*dts[i*2] - dion - dtrp - sys_bias;
        if (idx >= 0 && idx < NFREQ) pr_corr -= dant[idx];

        /* Do not apply Sagnac-rate correction here. Satellite clock drift is
         * still compensated; receiver clock drift remains for Teensy to estimate.
         */
        prrate_corr = -obs[i].D[0]*CLIGHT/freq + CLIGHT*dts[i*2+1];
        sig_name = tca_signal_name(sys,obs[i].code[0]);

        tca_buff_append(tca_rows_buff,&tca_rows_len,
            "%.6f,%.9f,"
            "%.6f,%.6f,%.6f,"
            "%.6f,%.6f,%.6f,"
            "%.12e,%.6f,%.2f,%s\n",
            pr_corr,
            prrate_corr,
            rs[i*6+0],rs[i*6+1],rs[i*6+2],
            rs[i*6+3],rs[i*6+4],rs[i*6+5],
            dts[i*2+1],
            azel_i[1]*R2D,
            cn0,
            sig_name);

        count++;
    }

    if (count<=0) {
        tca_export_reset();
        return;
    }

    tca_buff_append(tca_packet_buff,&tca_packet_len,
                    "TCA,%d,%.6f\n",count,time2gpst(obs[0].time,&week));
    tca_buff_append(tca_packet_buff,&tca_packet_len,"%s",tca_rows_buff);
    tca_buff_append(tca_packet_buff,&tca_packet_len,"END\n");
}


/* export TCA features directly from obs + nav ---------------------------------
 * This public function is intended to be called by sdr_pvt.c every epoch,
 * independent of PVT success/failure. It computes satellite position/clock from
 * nav, applies the same pre-fit corrections/checks used by the TCA exporter,
 * and writes the CSV packet returned by pntpos_get_tca_csv().
 *
 * rr_ref must be an approximate receiver ECEF position. In practice use the
 * last valid PVT position, a surveyed static position, or an INS-predicted ECEF
 * position. Without rr_ref, iono/tropo/elevation correction is not reliable.
 *
 * return: number of exported TCA rows. 0 means no packet was produced.
 *---------------------------------------------------------------------------*/
extern int pntpos_export_tca_features6(const obsd_t *obs, int n,
                                       const nav_t *nav,
                                       const prcopt_t *opt,
                                       const double *rr_ref,
                                       const double *rr_vel)
{
    prcopt_t opt_;
    double *rs=NULL,*dts=NULL,*var=NULL;
    int *svh=NULL;
    int rows=0;

    if (!obs || n<=0 || !nav || !opt || !rr_ref || norm(rr_ref,3)<=0.0) {
        tca_export_reset();
        return 0;
    }

    opt_=*opt;

    rs = mat(6,n);
    dts = mat(2,n);
    var = mat(1,n);
    svh = (int *)malloc(sizeof(int)*n);

    if (!rs || !dts || !var || !svh) {
        free(rs); free(dts); free(var); free(svh);
        tca_export_reset();
        return 0;
    }

    satposs(obs[0].time,obs,n,nav,opt_.sateph,rs,dts,var,svh);

    export_tca_csv_from_pntpos(obs,n,nav,&opt_,rr_ref,rr_vel,rs,dts,var,svh,
                               NULL,NULL,0); /* require_vsat=0 */

    if (tca_packet_buff[0]) {
        sscanf(tca_packet_buff,"TCA,%d",&rows);
    }

    free(rs); free(dts); free(var); free(svh);
    return rows;
}


/* Backward-compatible wrappers used by older sdr_pvt.c builds. --------------*/
extern int pntpos_export_tca_features(const obsd_t *obs, int n,
                                      const nav_t *nav,
                                      const prcopt_t *opt,
                                      const double *rr_ref)
{
    return pntpos_export_tca_features6(obs,n,nav,opt,rr_ref,NULL);
}

extern int pntpos_export_tca_feature(const obsd_t *obs, int n,
                                     const nav_t *nav,
                                     const prcopt_t *opt,
                                     const double *rr_ref)
{
    return pntpos_export_tca_features(obs,n,nav,opt,rr_ref);
}

/* single-point positioning ----------------------------------------------------
* compute receiver position, velocity, clock bias by single-point positioning
* with pseudorange and doppler observables
* args   : obsd_t *obs      I   observation data
*          int    n         I   number of observation data
*          nav_t  *nav      I   navigation data
*          prcopt_t *opt    I   processing options
*          sol_t  *sol      IO  solution
*          double *azel     IO  azimuth/elevation angle (rad) (NULL: no output)
*          ssat_t *ssat     IO  satellite status              (NULL: no output)
*          char   *msg      O   error message for error exit
* return : status(1:ok,0:error)
*-----------------------------------------------------------------------------*/
extern int pntpos(const obsd_t *obs, int n, const nav_t *nav,
                  const prcopt_t *opt, sol_t *sol, double *azel, ssat_t *ssat,
                  char *msg)
{
    prcopt_t opt_=*opt;
    double *rs,*dts,*var,*azel_,*resp;
    int i,stat,vsat[MAXOBS]={0},svh[MAXOBS];

    trace(3,"pntpos  : tobs=%s n=%d\n",time_str(obs[0].time,3),n);

    sol->stat=SOLQ_NONE;
    tca_export_reset();

    if (n<=0) {
        strcpy(msg,"no observation data");
        return 0;
    }
    sol->time=obs[0].time;
    sol->dtr[5]=0.0;
    msg[0]='\0';

    rs=mat(6,n); dts=mat(2,n); var=mat(1,n); azel_=zeros(2,n); resp=mat(1,n);

    if (opt_.mode!=PMODE_SINGLE) { /* for precise positioning */
        opt_.ionoopt=IONOOPT_BRDC;
        opt_.tropopt=TROPOPT_SAAS;
    }

    /* satellite positions, velocities and clocks */
    satposs(sol->time,obs,n,nav,opt_.sateph,rs,dts,var,svh);

    /* ----------------------------------------------------------------------
     * Independent TCA path before PVT estimation.
     *
     * This does not modify stat, sol, vsat, azel_, resp or ssat.
     * Therefore it does not disturb normal PocketSDR PVT generation.
     *
     * It can export TCA rows with only one valid satellite, but only after
     * at least one valid PVT has provided tca_ref_rr for iono/tropo/elevation.
     * --------------------------------------------------------------------*/
    if (tca_has_ref_rr) {
        export_tca_csv_from_pntpos(obs,n,nav,&opt_,tca_ref_rr,NULL,rs,dts,var,svh,
                                   NULL,NULL,0); /* require_vsat=0 */
    }

    /* estimate receiver position with pseudorange: normal PVT path unchanged */
    stat=estpos(obs,n,rs,dts,var,svh,nav,&opt_,sol,azel_,vsat,resp,msg);

    /* RAIM FDE. In addition to the original fail-only case, run one FDE pass
     * if the accepted solution still has a large post-fit residual. This
     * rejects a bad satellite instead of smoothing or moving sol->rr after the
     * fact. TCA compensation/export is not changed by this gate.
     */
    if (opt->posopt[4]&&n>=PNTPOS_RAIM_MIN_NSAT&&
        (!stat||need_residual_fde(vsat,resp,n))) {
        stat=raim_fde(obs,n,rs,dts,var,svh,nav,&opt_,sol,azel_,vsat,resp,msg);
    }

    /* estimate receiver velocity with Doppler */
    if (stat) {
        estvel(obs,n,rs,dts,nav,&opt_,sol,azel_,vsat);

        /* Update reference receiver position and inter-system biases for
         * future partial TCA epochs. dtr[0] is saved but not removed in the
         * TCA exporter; only dtr[1..4] are used for ISB compensation.
         */
        tca_ref_rr[0]=sol->rr[0];
        tca_ref_rr[1]=sol->rr[1];
        tca_ref_rr[2]=sol->rr[2];
        tca_has_ref_rr=1;
        for (i=0;i<5;i++) tca_ref_dtr[i]=sol->dtr[i];
        tca_has_ref_dtr=1;

        /* In locked PVT mode, overwrite with PVT/RAIM accepted rows. */
        export_tca_csv_from_pntpos(obs,n,nav,&opt_,sol->rr,sol->rr+3,rs,dts,var,svh,
                                   azel_,vsat,1); /* require_vsat=1 */
    }
    /* If stat==0, do not reset here. The independent TCA path above may have
     * already produced a valid TCA,1... packet using tca_ref_rr.
     */

    if (azel) {
        for (i=0;i<n*2;i++) azel[i]=azel_[i];
    }
    if (ssat) {
        for (i=0;i<MAXSAT;i++) {
            ssat[i].vs=0;
            ssat[i].azel[0]=ssat[i].azel[1]=0.0;
            ssat[i].resp[0]=ssat[i].resc[0]=0.0;
            ssat[i].snr[0]=0;
        }
        for (i=0;i<n;i++) {
            if (obs[i].sat<=0||obs[i].sat>MAXSAT) continue;
            ssat[obs[i].sat-1].azel[0]=azel_[  i*2];
            ssat[obs[i].sat-1].azel[1]=azel_[1+i*2];
            ssat[obs[i].sat-1].snr[0]=obs[i].SNR[0];
            if (!vsat[i]) continue;
            ssat[obs[i].sat-1].vs=1;
            ssat[obs[i].sat-1].resp[0]=resp[i];
        }
    }
    free(rs); free(dts); free(var); free(azel_); free(resp);
    return stat;
}
