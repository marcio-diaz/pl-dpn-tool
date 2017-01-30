/*
 * Allwinner A1X SoCs pinctrl driver.
 *
 * Copyright (C) 2012 Maxime Ripard
 *
 * Maxime Ripard <maxime.ripard@free-electrons.com>
 *
 * This file is licensed under the terms of the GNU General Public
 * License version 2.  This program is licensed "as is" without any
 * warranty of any kind, whether express or implied.
 */

#ifndef __PINCTRL_SUNXI_H
#define __PINCTRL_SUNXI_H

#include <linux/kernel.h>
#include <linux/spinlock.h>

#define PA_BASE	0
#define PB_BASE	32
#define PC_BASE	64
#define PD_BASE	96
#define PE_BASE	128
#define PF_BASE	160
#define PG_BASE	192
#define PH_BASE	224
#define PI_BASE	256

#define SUNXI_PINCTRL_PIN_PA0	PINCTRL_PIN(PA_BASE + 0, "PA0")
#define SUNXI_PINCTRL_PIN_PA1	PINCTRL_PIN(PA_BASE + 1, "PA1")
#define SUNXI_PINCTRL_PIN_PA2	PINCTRL_PIN(PA_BASE + 2, "PA2")
#define SUNXI_PINCTRL_PIN_PA3	PINCTRL_PIN(PA_BASE + 3, "PA3")
#define SUNXI_PINCTRL_PIN_PA4	PINCTRL_PIN(PA_BASE + 4, "PA4")
#define SUNXI_PINCTRL_PIN_PA5	PINCTRL_PIN(PA_BASE + 5, "PA5")
#define SUNXI_PINCTRL_PIN_PA6	PINCTRL_PIN(PA_BASE + 6, "PA6")
#define SUNXI_PINCTRL_PIN_PA7	PINCTRL_PIN(PA_BASE + 7, "PA7")
#define SUNXI_PINCTRL_PIN_PA8	PINCTRL_PIN(PA_BASE + 8, "PA8")
#define SUNXI_PINCTRL_PIN_PA9	PINCTRL_PIN(PA_BASE + 9, "PA9")
#define SUNXI_PINCTRL_PIN_PA10	PINCTRL_PIN(PA_BASE + 10, "PA10")
#define SUNXI_PINCTRL_PIN_PA11	PINCTRL_PIN(PA_BASE + 11, "PA11")
#define SUNXI_PINCTRL_PIN_PA12	PINCTRL_PIN(PA_BASE + 12, "PA12")
#define SUNXI_PINCTRL_PIN_PA13	PINCTRL_PIN(PA_BASE + 13, "PA13")
#define SUNXI_PINCTRL_PIN_PA14	PINCTRL_PIN(PA_BASE + 14, "PA14")
#define SUNXI_PINCTRL_PIN_PA15	PINCTRL_PIN(PA_BASE + 15, "PA15")
#define SUNXI_PINCTRL_PIN_PA16	PINCTRL_PIN(PA_BASE + 16, "PA16")
#define SUNXI_PINCTRL_PIN_PA17	PINCTRL_PIN(PA_BASE + 17, "PA17")
#define SUNXI_PINCTRL_PIN_PA18	PINCTRL_PIN(PA_BASE + 18, "PA18")
#define SUNXI_PINCTRL_PIN_PA19	PINCTRL_PIN(PA_BASE + 19, "PA19")
#define SUNXI_PINCTRL_PIN_PA20	PINCTRL_PIN(PA_BASE + 20, "PA20")
#define SUNXI_PINCTRL_PIN_PA21	PINCTRL_PIN(PA_BASE + 21, "PA21")
#define SUNXI_PINCTRL_PIN_PA22	PINCTRL_PIN(PA_BASE + 22, "PA22")
#define SUNXI_PINCTRL_PIN_PA23	PINCTRL_PIN(PA_BASE + 23, "PA23")
#define SUNXI_PINCTRL_PIN_PA24	PINCTRL_PIN(PA_BASE + 24, "PA24")
#define SUNXI_PINCTRL_PIN_PA25	PINCTRL_PIN(PA_BASE + 25, "PA25")
#define SUNXI_PINCTRL_PIN_PA26	PINCTRL_PIN(PA_BASE + 26, "PA26")
#define SUNXI_PINCTRL_PIN_PA27	PINCTRL_PIN(PA_BASE + 27, "PA27")
#define SUNXI_PINCTRL_PIN_PA28	PINCTRL_PIN(PA_BASE + 28, "PA28")
#define SUNXI_PINCTRL_PIN_PA29	PINCTRL_PIN(PA_BASE + 29, "PA29")
#define SUNXI_PINCTRL_PIN_PA30	PINCTRL_PIN(PA_BASE + 30, "PA30")
#define SUNXI_PINCTRL_PIN_PA31	PINCTRL_PIN(PA_BASE + 31, "PA31")

#define SUNXI_PINCTRL_PIN_PB0	PINCTRL_PIN(PB_BASE + 0, "PB0")
#define SUNXI_PINCTRL_PIN_PB1	PINCTRL_PIN(PB_BASE + 1, "PB1")
#define SUNXI_PINCTRL_PIN_PB2	PINCTRL_PIN(PB_BASE + 2, "PB2")
#define SUNXI_PINCTRL_PIN_PB3	PINCTRL_PIN(PB_BASE + 3, "PB3")
#define SUNXI_PINCTRL_PIN_PB4	PINCTRL_PIN(PB_BASE + 4, "PB4")
#define SUNXI_PINCTRL_PIN_PB5	PINCTRL_PIN(PB_BASE + 5, "PB5")
#define SUNXI_PINCTRL_PIN_PB6	PINCTRL_PIN(PB_BASE + 6, "PB6")
#define SUNXI_PINCTRL_PIN_PB7	PINCTRL_PIN(PB_BASE + 7, "PB7")
#define SUNXI_PINCTRL_PIN_PB8	PINCTRL_PIN(PB_BASE + 8, "PB8")
#define SUNXI_PINCTRL_PIN_PB9	PINCTRL_PIN(PB_BASE + 9, "PB9")
#define SUNXI_PINCTRL_PIN_PB10	PINCTRL_PIN(PB_BASE + 10, "PB10")
#define SUNXI_PINCTRL_PIN_PB11	PINCTRL_PIN(PB_BASE + 11, "PB11")
#define SUNXI_PINCTRL_PIN_PB12	PINCTRL_PIN(PB_BASE + 12, "PB12")
#define SUNXI_PINCTRL_PIN_PB13	PINCTRL_PIN(PB_BASE + 13, "PB13")
#define SUNXI_PINCTRL_PIN_PB14	PINCTRL_PIN(PB_BASE + 14, "PB14")
#define SUNXI_PINCTRL_PIN_PB15	PINCTRL_PIN(PB_BASE + 15, "PB15")
#define SUNXI_PINCTRL_PIN_PB16	PINCTRL_PIN(PB_BASE + 16, "PB16")
#define SUNXI_PINCTRL_PIN_PB17	PINCTRL_PIN(PB_BASE + 17, "PB17")
#define SUNXI_PINCTRL_PIN_PB18	PINCTRL_PIN(PB_BASE + 18, "PB18")
#define SUNXI_PINCTRL_PIN_PB19	PINCTRL_PIN(PB_BASE + 19, "PB19")
#define SUNXI_PINCTRL_PIN_PB20	PINCTRL_PIN(PB_BASE + 20, "PB20")
#define SUNXI_PINCTRL_PIN_PB21	PINCTRL_PIN(PB_BASE + 21, "PB21")
#define SUNXI_PINCTRL_PIN_PB22	PINCTRL_PIN(PB_BASE + 22, "PB22")
#define SUNXI_PINCTRL_PIN_PB23	PINCTRL_PIN(PB_BASE + 23, "PB23")
#define SUNXI_PINCTRL_PIN_PB24	PINCTRL_PIN(PB_BASE + 24, "PB24")
#define SUNXI_PINCTRL_PIN_PB25	PINCTRL_PIN(PB_BASE + 25, "PB25")
#define SUNXI_PINCTRL_PIN_PB26	PINCTRL_PIN(PB_BASE + 26, "PB26")
#define SUNXI_PINCTRL_PIN_PB27	PINCTRL_PIN(PB_BASE + 27, "PB27")
#define SUNXI_PINCTRL_PIN_PB28	PINCTRL_PIN(PB_BASE + 28, "PB28")
#define SUNXI_PINCTRL_PIN_PB29	PINCTRL_PIN(PB_BASE + 29, "PB29")
#define SUNXI_PINCTRL_PIN_PB30	PINCTRL_PIN(PB_BASE + 30, "PB30")
#define SUNXI_PINCTRL_PIN_PB31	PINCTRL_PIN(PB_BASE + 31, "PB31")

#define SUNXI_PINCTRL_PIN_PC0	PINCTRL_PIN(PC_BASE + 0, "PC0")
#define SUNXI_PINCTRL_PIN_PC1	PINCTRL_PIN(PC_BASE + 1, "PC1")
#define SUNXI_PINCTRL_PIN_PC2	PINCTRL_PIN(PC_BASE + 2, "PC2")
#define SUNXI_PINCTRL_PIN_PC3	PINCTRL_PIN(PC_BASE + 3, "PC3")
#define SUNXI_PINCTRL_PIN_PC4	PINCTRL_PIN(PC_BASE + 4, "PC4")
#define SUNXI_PINCTRL_PIN_PC5	PINCTRL_PIN(PC_BASE + 5, "PC5")
#define SUNXI_PINCTRL_PIN_PC6	PINCTRL_PIN(PC_BASE + 6, "PC6")
#define SUNXI_PINCTRL_PIN_PC7	PINCTRL_PIN(PC_BASE + 7, "PC7")
#define SUNXI_PINCTRL_PIN_PC8	PINCTRL_PIN(PC_BASE + 8, "PC8")
#define SUNXI_PINCTRL_PIN_PC9	PINCTRL_PIN(PC_BASE + 9, "PC9")
#define SUNXI_PINCTRL_PIN_PC10	PINCTRL_PIN(PC_BASE + 10, "PC10")
#define SUNXI_PINCTRL_PIN_PC11	PINCTRL_PIN(PC_BASE + 11, "PC11")
#define SUNXI_PINCTRL_PIN_PC12	PINCTRL_PIN(PC_BASE + 12, "PC12")
#define SUNXI_PINCTRL_PIN_PC13	PINCTRL_PIN(PC_BASE + 13, "PC13")
#define SUNXI_PINCTRL_PIN_PC14	PINCTRL_PIN(PC_BASE + 14, "PC14")
#define SUNXI_PINCTRL_PIN_PC15	PINCTRL_PIN(PC_BASE + 15, "PC15")
#define SUNXI_PINCTRL_PIN_PC16	PINCTRL_PIN(PC_BASE + 16, "PC16")
#define SUNXI_PINCTRL_PIN_PC17	PINCTRL_PIN(PC_BASE + 17, "PC17")
#define SUNXI_PINCTRL_PIN_PC18	PINCTRL_PIN(PC_BASE + 18, "PC18")
#define SUNXI_PINCTRL_PIN_PC19	PINCTRL_PIN(PC_BASE + 19, "PC19")
#define SUNXI_PINCTRL_PIN_PC20	PINCTRL_PIN(PC_BASE + 20, "PC20")
#define SUNXI_PINCTRL_PIN_PC21	PINCTRL_PIN(PC_BASE + 21, "PC21")
#define SUNXI_PINCTRL_PIN_PC22	PINCTRL_PIN(PC_BASE + 22, "PC22")
#define SUNXI_PINCTRL_PIN_PC23	PINCTRL_PIN(PC_BASE + 23, "PC23")
#define SUNXI_PINCTRL_PIN_PC24	PINCTRL_PIN(PC_BASE + 24, "PC24")
#define SUNXI_PINCTRL_PIN_PC25	PINCTRL_PIN(PC_BASE + 25, "PC25")
#define SUNXI_PINCTRL_PIN_PC26	PINCTRL_PIN(PC_BASE + 26, "PC26")
#define SUNXI_PINCTRL_PIN_PC27	PINCTRL_PIN(PC_BASE + 27, "PC27")
#define SUNXI_PINCTRL_PIN_PC28	PINCTRL_PIN(PC_BASE + 28, "PC28")
#define SUNXI_PINCTRL_PIN_PC29	PINCTRL_PIN(PC_BASE + 29, "PC29")
#define SUNXI_PINCTRL_PIN_PC30	PINCTRL_PIN(PC_BASE + 30, "PC30")
#define SUNXI_PINCTRL_PIN_PC31	PINCTRL_PIN(PC_BASE + 31, "PC31")

#define SUNXI_PINCTRL_PIN_PD0	PINCTRL_PIN(PD_BASE + 0, "PD0")
#define SUNXI_PINCTRL_PIN_PD1	PINCTRL_PIN(PD_BASE + 1, "PD1")
#define SUNXI_PINCTRL_PIN_PD2	PINCTRL_PIN(PD_BASE + 2, "PD2")
#define SUNXI_PINCTRL_PIN_PD3	PINCTRL_PIN(PD_BASE + 3, "PD3")
#define SUNXI_PINCTRL_PIN_PD4	PINCTRL_PIN(PD_BASE + 4, "PD4")
#define SUNXI_PINCTRL_PIN_PD5	PINCTRL_PIN(PD_BASE + 5, "PD5")
#define SUNXI_PINCTRL_PIN_PD6	PINCTRL_PIN(PD_BASE + 6, "PD6")
#define SUNXI_PINCTRL_PIN_PD7	PINCTRL_PIN(PD_BASE + 7, "PD7")
#define SUNXI_PINCTRL_PIN_PD8	PINCTRL_PIN(PD_BASE + 8, "PD8")
#define SUNXI_PINCTRL_PIN_PD9	PINCTRL_PIN(PD_BASE + 9, "PD9")
#define SUNXI_PINCTRL_PIN_PD10	PINCTRL_PIN(PD_BASE + 10, "PD10")
#define SUNXI_PINCTRL_PIN_PD11	PINCTRL_PIN(PD_BASE + 11, "PD11")
#define SUNXI_PINCTRL_PIN_PD12	PINCTRL_PIN(PD_BASE + 12, "PD12")
#define SUNXI_PINCTRL_PIN_PD13	PINCTRL_PIN(PD_BASE + 13, "PD13")
#define SUNXI_PINCTRL_PIN_PD14	PINCTRL_PIN(PD_BASE + 14, "PD14")
#define SUNXI_PINCTRL_PIN_PD15	PINCTRL_PIN(PD_BASE + 15, "PD15")
#define SUNXI_PINCTRL_PIN_PD16	PINCTRL_PIN(PD_BASE + 16, "PD16")
#define SUNXI_PINCTRL_PIN_PD17	PINCTRL_PIN(PD_BASE + 17, "PD17")
#define SUNXI_PINCTRL_PIN_PD18	PINCTRL_PIN(PD_BASE + 18, "PD18")
#define SUNXI_PINCTRL_PIN_PD19	PINCTRL_PIN(PD_BASE + 19, "PD19")
#define SUNXI_PINCTRL_PIN_PD20	PINCTRL_PIN(PD_BASE + 20, "PD20")
#define SUNXI_PINCTRL_PIN_PD21	PINCTRL_PIN(PD_BASE + 21, "PD21")
#define SUNXI_PINCTRL_PIN_PD22	PINCTRL_PIN(PD_BASE + 22, "PD22")
#define SUNXI_PINCTRL_PIN_PD23	PINCTRL_PIN(PD_BASE + 23, "PD23")
#define SUNXI_PINCTRL_PIN_PD24	PINCTRL_PIN(PD_BASE + 24, "PD24")
#define SUNXI_PINCTRL_PIN_PD25	PINCTRL_PIN(PD_BASE + 25, "PD25")
#define SUNXI_PINCTRL_PIN_PD26	PINCTRL_PIN(PD_BASE + 26, "PD26")
#define SUNXI_PINCTRL_PIN_PD27	PINCTRL_PIN(PD_BASE + 27, "PD27")
#define SUNXI_PINCTRL_PIN_PD28	PINCTRL_PIN(PD_BASE + 28, "PD28")
#define SUNXI_PINCTRL_PIN_PD29	PINCTRL_PIN(PD_BASE + 29, "PD29")
#define SUNXI_PINCTRL_PIN_PD30	PINCTRL_PIN(PD_BASE + 30, "PD30")
#define SUNXI_PINCTRL_PIN_PD31	PINCTRL_PIN(PD_BASE + 31, "PD31")

#define SUNXI_PINCTRL_PIN_PE0	PINCTRL_PIN(PE_BASE + 0, "PE0")
#define SUNXI_PINCTRL_PIN_PE1	PINCTRL_PIN(PE_BASE + 1, "PE1")
#define SUNXI_PINCTRL_PIN_PE2	PINCTRL_PIN(PE_BASE + 2, "PE2")
#define SUNXI_PINCTRL_PIN_PE3	PINCTRL_PIN(PE_BASE + 3, "PE3")
#define SUNXI_PINCTRL_PIN_PE4	PINCTRL_PIN(PE_BASE + 4, "PE4")
#define SUNXI_PINCTRL_PIN_PE5	PINCTRL_PIN(PE_BASE + 5, "PE5")
#define SUNXI_PINCTRL_PIN_PE6	PINCTRL_PIN(PE_BASE + 6, "PE6")
#define SUNXI_PINCTRL_PIN_PE7	PINCTRL_PIN(PE_BASE + 7, "PE7")
#define SUNXI_PINCTRL_PIN_PE8	PINCTRL_PIN(PE_BASE + 8, "PE8")
#define SUNXI_PINCTRL_PIN_PE9	PINCTRL_PIN(PE_BASE + 9, "PE9")
#define SUNXI_PINCTRL_PIN_PE10	PINCTRL_PIN(PE_BASE + 10, "PE10")
#define SUNXI_PINCTRL_PIN_PE11	PINCTRL_PIN(PE_BASE + 11, "PE11")
#define SUNXI_PINCTRL_PIN_PE12	PINCTRL_PIN(PE_BASE + 12, "PE12")
#define SUNXI_PINCTRL_PIN_PE13	PINCTRL_PIN(PE_BASE + 13, "PE13")
#define SUNXI_PINCTRL_PIN_PE14	PINCTRL_PIN(PE_BASE + 14, "PE14")
#define SUNXI_PINCTRL_PIN_PE15	PINCTRL_PIN(PE_BASE + 15, "PE15")
#define SUNXI_PINCTRL_PIN_PE16	PINCTRL_PIN(PE_BASE + 16, "PE16")
#define SUNXI_PINCTRL_PIN_PE17	PINCTRL_PIN(PE_BASE + 17, "PE17")
#define SUNXI_PINCTRL_PIN_PE18	PINCTRL_PIN(PE_BASE + 18, "PE18")
#define SUNXI_PINCTRL_PIN_PE19	PINCTRL_PIN(PE_BASE + 19, "PE19")
#define SUNXI_PINCTRL_PIN_PE20	PINCTRL_PIN(PE_BASE + 20, "PE20")
#define SUNXI_PINCTRL_PIN_PE21	PINCTRL_PIN(PE_BASE + 21, "PE21")
#define SUNXI_PINCTRL_PIN_PE22	PINCTRL_PIN(PE_BASE + 22, "PE22")
#define SUNXI_PINCTRL_PIN_PE23	PINCTRL_PIN(PE_BASE + 23, "PE23")
#define SUNXI_PINCTRL_PIN_PE24	PINCTRL_PIN(PE_BASE + 24, "PE24")
#define SUNXI_PINCTRL_PIN_PE25	PINCTRL_PIN(PE_BASE + 25, "PE25")
#define SUNXI_PINCTRL_PIN_PE26	PINCTRL_PIN(PE_BASE + 26, "PE26")
#define SUNXI_PINCTRL_PIN_PE27	PINCTRL_PIN(PE_BASE + 27, "PE27")
#define SUNXI_PINCTRL_PIN_PE28	PINCTRL_PIN(PE_BASE + 28, "PE28")
#define SUNXI_PINCTRL_PIN_PE29	PINCTRL_PIN(PE_BASE + 29, "PE29")
#define SUNXI_PINCTRL_PIN_PE30	PINCTRL_PIN(PE_BASE + 30, "PE30")
#define SUNXI_PINCTRL_PIN_PE31	PINCTRL_PIN(PE_BASE + 31, "PE31")

#define SUNXI_PINCTRL_PIN_PF0	PINCTRL_PIN(PF_BASE + 0, "PF0")
#define SUNXI_PINCTRL_PIN_PF1	PINCTRL_PIN(PF_BASE + 1, "PF1")
#define SUNXI_PINCTRL_PIN_PF2	PINCTRL_PIN(PF_BASE + 2, "PF2")
#define SUNXI_PINCTRL_PIN_PF3	PINCTRL_PIN(PF_BASE + 3, "PF3")
#define SUNXI_PINCTRL_PIN_PF4	PINCTRL_PIN(PF_BASE + 4, "PF4")
#define SUNXI_PINCTRL_PIN_PF5	PINCTRL_PIN(PF_BASE + 5, "PF5")
#define SUNXI_PINCTRL_PIN_PF6	PINCTRL_PIN(PF_BASE + 6, "PF6")
#define SUNXI_PINCTRL_PIN_PF7	PINCTRL_PIN(PF_BASE + 7, "PF7")
#define SUNXI_PINCTRL_PIN_PF8	PINCTRL_PIN(PF_BASE + 8, "PF8")
#define SUNXI_PINCTRL_PIN_PF9	PINCTRL_PIN(PF_BASE + 9, "PF9")
#define SUNXI_PINCTRL_PIN_PF10	PINCTRL_PIN(PF_BASE + 10, "PF10")
#define SUNXI_PINCTRL_PIN_PF11	PINCTRL_PIN(PF_BASE + 11, "PF11")
#define SUNXI_PINCTRL_PIN_PF12	PINCTRL_PIN(PF_BASE + 12, "PF12")
#define SUNXI_PINCTRL_PIN_PF13	PINCTRL_PIN(PF_BASE + 13, "PF13")
#define SUNXI_PINCTRL_PIN_PF14	PINCTRL_PIN(PF_BASE + 14, "PF14")
#define SUNXI_PINCTRL_PIN_PF15	PINCTRL_PIN(PF_BASE + 15, "PF15")
#define SUNXI_PINCTRL_PIN_PF16	PINCTRL_PIN(PF_BASE + 16, "PF16")
#define SUNXI_PINCTRL_PIN_PF17	PINCTRL_PIN(PF_BASE + 17, "PF17")
#define SUNXI_PINCTRL_PIN_PF18	PINCTRL_PIN(PF_BASE + 18, "PF18")
#define SUNXI_PINCTRL_PIN_PF19	PINCTRL_PIN(PF_BASE + 19, "PF19")
#define SUNXI_PINCTRL_PIN_PF20	PINCTRL_PIN(PF_BASE + 20, "PF20")
#define SUNXI_PINCTRL_PIN_PF21	PINCTRL_PIN(PF_BASE + 21, "PF21")
#define SUNXI_PINCTRL_PIN_PF22	PINCTRL_PIN(PF_BASE + 22, "PF22")
#define SUNXI_PINCTRL_PIN_PF23	PINCTRL_PIN(PF_BASE + 23, "PF23")
#define SUNXI_PINCTRL_PIN_PF24	PINCTRL_PIN(PF_BASE + 24, "PF24")
#define SUNXI_PINCTRL_PIN_PF25	PINCTRL_PIN(PF_BASE + 25, "PF25")
#define SUNXI_PINCTRL_PIN_PF26	PINCTRL_PIN(PF_BASE + 26, "PF26")
#define SUNXI_PINCTRL_PIN_PF27	PINCTRL_PIN(PF_BASE + 27, "PF27")
#define SUNXI_PINCTRL_PIN_PF28	PINCTRL_PIN(PF_BASE + 28, "PF28")
#define SUNXI_PINCTRL_PIN_PF29	PINCTRL_PIN(PF_BASE + 29, "PF29")
#define SUNXI_PINCTRL_PIN_PF30	PINCTRL_PIN(PF_BASE + 30, "PF30")
#define SUNXI_PINCTRL_PIN_PF31	PINCTRL_PIN(PF_BASE + 31, "PF31")

#define SUNXI_PINCTRL_PIN_PG0	PINCTRL_PIN(PG_BASE + 0, "PG0")
#define SUNXI_PINCTRL_PIN_PG1	PINCTRL_PIN(PG_BASE + 1, "PG1")
#define SUNXI_PINCTRL_PIN_PG2	PINCTRL_PIN(PG_BASE + 2, "PG2")
#define SUNXI_PINCTRL_PIN_PG3	PINCTRL_PIN(PG_BASE + 3, "PG3")
#define SUNXI_PINCTRL_PIN_PG4	PINCTRL_PIN(PG_BASE + 4, "PG4")
#define SUNXI_PINCTRL_PIN_PG5	PINCTRL_PIN(PG_BASE + 5, "PG5")
#define SUNXI_PINCTRL_PIN_PG6	PINCTRL_PIN(PG_BASE + 6, "PG6")
#define SUNXI_PINCTRL_PIN_PG7	PINCTRL_PIN(PG_BASE + 7, "PG7")
#define SUNXI_PINCTRL_PIN_PG8	PINCTRL_PIN(PG_BASE + 8, "PG8")
#define SUNXI_PINCTRL_PIN_PG9	PINCTRL_PIN(PG_BASE + 9, "PG9")
#define SUNXI_PINCTRL_PIN_PG10	PINCTRL_PIN(PG_BASE + 10, "PG10")
#define SUNXI_PINCTRL_PIN_PG11	PINCTRL_PIN(PG_BASE + 11, "PG11")
#define SUNXI_PINCTRL_PIN_PG12	PINCTRL_PIN(PG_BASE + 12, "PG12")
#define SUNXI_PINCTRL_PIN_PG13	PINCTRL_PIN(PG_BASE + 13, "PG13")
#define SUNXI_PINCTRL_PIN_PG14	PINCTRL_PIN(PG_BASE + 14, "PG14")
#define SUNXI_PINCTRL_PIN_PG15	PINCTRL_PIN(PG_BASE + 15, "PG15")
#define SUNXI_PINCTRL_PIN_PG16	PINCTRL_PIN(PG_BASE + 16, "PG16")
#define SUNXI_PINCTRL_PIN_PG17	PINCTRL_PIN(PG_BASE + 17, "PG17")
#define SUNXI_PINCTRL_PIN_PG18	PINCTRL_PIN(PG_BASE + 18, "PG18")
#define SUNXI_PINCTRL_PIN_PG19	PINCTRL_PIN(PG_BASE + 19, "PG19")
#define SUNXI_PINCTRL_PIN_PG20	PINCTRL_PIN(PG_BASE + 20, "PG20")
#define SUNXI_PINCTRL_PIN_PG21	PINCTRL_PIN(PG_BASE + 21, "PG21")
#define SUNXI_PINCTRL_PIN_PG22	PINCTRL_PIN(PG_BASE + 22, "PG22")
#define SUNXI_PINCTRL_PIN_PG23	PINCTRL_PIN(PG_BASE + 23, "PG23")
#define SUNXI_PINCTRL_PIN_PG24	PINCTRL_PIN(PG_BASE + 24, "PG24")
#define SUNXI_PINCTRL_PIN_PG25	PINCTRL_PIN(PG_BASE + 25, "PG25")
#define SUNXI_PINCTRL_PIN_PG26	PINCTRL_PIN(PG_BASE + 26, "PG26")
#define SUNXI_PINCTRL_PIN_PG27	PINCTRL_PIN(PG_BASE + 27, "PG27")
#define SUNXI_PINCTRL_PIN_PG28	PINCTRL_PIN(PG_BASE + 28, "PG28")
#define SUNXI_PINCTRL_PIN_PG29	PINCTRL_PIN(PG_BASE + 29, "PG29")
#define SUNXI_PINCTRL_PIN_PG30	PINCTRL_PIN(PG_BASE + 30, "PG30")
#define SUNXI_PINCTRL_PIN_PG31	PINCTRL_PIN(PG_BASE + 31, "PG31")

#define SUNXI_PINCTRL_PIN_PH0	PINCTRL_PIN(PH_BASE + 0, "PH0")
#define SUNXI_PINCTRL_PIN_PH1	PINCTRL_PIN(PH_BASE + 1, "PH1")
#define SUNXI_PINCTRL_PIN_PH2	PINCTRL_PIN(PH_BASE + 2, "PH2")
#define SUNXI_PINCTRL_PIN_PH3	PINCTRL_PIN(PH_BASE + 3, "PH3")
#define SUNXI_PINCTRL_PIN_PH4	PINCTRL_PIN(PH_BASE + 4, "PH4")
#define SUNXI_PINCTRL_PIN_PH5	PINCTRL_PIN(PH_BASE + 5, "PH5")
#define SUNXI_PINCTRL_PIN_PH6	PINCTRL_PIN(PH_BASE + 6, "PH6")
#define SUNXI_PINCTRL_PIN_PH7	PINCTRL_PIN(PH_BASE + 7, "PH7")
#define SUNXI_PINCTRL_PIN_PH8	PINCTRL_PIN(PH_BASE + 8, "PH8")
#define SUNXI_PINCTRL_PIN_PH9	PINCTRL_PIN(PH_BASE + 9, "PH9")
#define SUNXI_PINCTRL_PIN_PH10	PINCTRL_PIN(PH_BASE + 10, "PH10")
#define SUNXI_PINCTRL_PIN_PH11	PINCTRL_PIN(PH_BASE + 11, "PH11")
#define SUNXI_PINCTRL_PIN_PH12	PINCTRL_PIN(PH_BASE + 12, "PH12")
#define SUNXI_PINCTRL_PIN_PH13	PINCTRL_PIN(PH_BASE + 13, "PH13")
#define SUNXI_PINCTRL_PIN_PH14	PINCTRL_PIN(PH_BASE + 14, "PH14")
#define SUNXI_PINCTRL_PIN_PH15	PINCTRL_PIN(PH_BASE + 15, "PH15")
#define SUNXI_PINCTRL_PIN_PH16	PINCTRL_PIN(PH_BASE + 16, "PH16")
#define SUNXI_PINCTRL_PIN_PH17	PINCTRL_PIN(PH_BASE + 17, "PH17")
#define SUNXI_PINCTRL_PIN_PH18	PINCTRL_PIN(PH_BASE + 18, "PH18")
#define SUNXI_PINCTRL_PIN_PH19	PINCTRL_PIN(PH_BASE + 19, "PH19")
#define SUNXI_PINCTRL_PIN_PH20	PINCTRL_PIN(PH_BASE + 20, "PH20")
#define SUNXI_PINCTRL_PIN_PH21	PINCTRL_PIN(PH_BASE + 21, "PH21")
#define SUNXI_PINCTRL_PIN_PH22	PINCTRL_PIN(PH_BASE + 22, "PH22")
#define SUNXI_PINCTRL_PIN_PH23	PINCTRL_PIN(PH_BASE + 23, "PH23")
#define SUNXI_PINCTRL_PIN_PH24	PINCTRL_PIN(PH_BASE + 24, "PH24")
#define SUNXI_PINCTRL_PIN_PH25	PINCTRL_PIN(PH_BASE + 25, "PH25")
#define SUNXI_PINCTRL_PIN_PH26	PINCTRL_PIN(PH_BASE + 26, "PH26")
#define SUNXI_PINCTRL_PIN_PH27	PINCTRL_PIN(PH_BASE + 27, "PH27")
#define SUNXI_PINCTRL_PIN_PH28	PINCTRL_PIN(PH_BASE + 28, "PH28")
#define SUNXI_PINCTRL_PIN_PH29	PINCTRL_PIN(PH_BASE + 29, "PH29")
#define SUNXI_PINCTRL_PIN_PH30	PINCTRL_PIN(PH_BASE + 30, "PH30")
#define SUNXI_PINCTRL_PIN_PH31	PINCTRL_PIN(PH_BASE + 31, "PH31")

#define SUNXI_PINCTRL_PIN_PI0	PINCTRL_PIN(PI_BASE + 0, "PI0")
#define SUNXI_PINCTRL_PIN_PI1	PINCTRL_PIN(PI_BASE + 1, "PI1")
#define SUNXI_PINCTRL_PIN_PI2	PINCTRL_PIN(PI_BASE + 2, "PI2")
#define SUNXI_PINCTRL_PIN_PI3	PINCTRL_PIN(PI_BASE + 3, "PI3")
#define SUNXI_PINCTRL_PIN_PI4	PINCTRL_PIN(PI_BASE + 4, "PI4")
#define SUNXI_PINCTRL_PIN_PI5	PINCTRL_PIN(PI_BASE + 5, "PI5")
#define SUNXI_PINCTRL_PIN_PI6	PINCTRL_PIN(PI_BASE + 6, "PI6")
#define SUNXI_PINCTRL_PIN_PI7	PINCTRL_PIN(PI_BASE + 7, "PI7")
#define SUNXI_PINCTRL_PIN_PI8	PINCTRL_PIN(PI_BASE + 8, "PI8")
#define SUNXI_PINCTRL_PIN_PI9	PINCTRL_PIN(PI_BASE + 9, "PI9")
#define SUNXI_PINCTRL_PIN_PI10	PINCTRL_PIN(PI_BASE + 10, "PI10")
#define SUNXI_PINCTRL_PIN_PI11	PINCTRL_PIN(PI_BASE + 11, "PI11")
#define SUNXI_PINCTRL_PIN_PI12	PINCTRL_PIN(PI_BASE + 12, "PI12")
#define SUNXI_PINCTRL_PIN_PI13	PINCTRL_PIN(PI_BASE + 13, "PI13")
#define SUNXI_PINCTRL_PIN_PI14	PINCTRL_PIN(PI_BASE + 14, "PI14")
#define SUNXI_PINCTRL_PIN_PI15	PINCTRL_PIN(PI_BASE + 15, "PI15")
#define SUNXI_PINCTRL_PIN_PI16	PINCTRL_PIN(PI_BASE + 16, "PI16")
#define SUNXI_PINCTRL_PIN_PI17	PINCTRL_PIN(PI_BASE + 17, "PI17")
#define SUNXI_PINCTRL_PIN_PI18	PINCTRL_PIN(PI_BASE + 18, "PI18")
#define SUNXI_PINCTRL_PIN_PI19	PINCTRL_PIN(PI_BASE + 19, "PI19")
#define SUNXI_PINCTRL_PIN_PI20	PINCTRL_PIN(PI_BASE + 20, "PI20")
#define SUNXI_PINCTRL_PIN_PI21	PINCTRL_PIN(PI_BASE + 21, "PI21")
#define SUNXI_PINCTRL_PIN_PI22	PINCTRL_PIN(PI_BASE + 22, "PI22")
#define SUNXI_PINCTRL_PIN_PI23	PINCTRL_PIN(PI_BASE + 23, "PI23")
#define SUNXI_PINCTRL_PIN_PI24	PINCTRL_PIN(PI_BASE + 24, "PI24")
#define SUNXI_PINCTRL_PIN_PI25	PINCTRL_PIN(PI_BASE + 25, "PI25")
#define SUNXI_PINCTRL_PIN_PI26	PINCTRL_PIN(PI_BASE + 26, "PI26")
#define SUNXI_PINCTRL_PIN_PI27	PINCTRL_PIN(PI_BASE + 27, "PI27")
#define SUNXI_PINCTRL_PIN_PI28	PINCTRL_PIN(PI_BASE + 28, "PI28")
#define SUNXI_PINCTRL_PIN_PI29	PINCTRL_PIN(PI_BASE + 29, "PI29")
#define SUNXI_PINCTRL_PIN_PI30	PINCTRL_PIN(PI_BASE + 30, "PI30")
#define SUNXI_PINCTRL_PIN_PI31	PINCTRL_PIN(PI_BASE + 31, "PI31")

#define SUNXI_PIN_NAME_MAX_LEN	5

#define BANK_MEM_SIZE		0x24
#define MUX_REGS_OFFSET		0x0
#define DATA_REGS_OFFSET	0x10
#define DLEVEL_REGS_OFFSET	0x14
#define PULL_REGS_OFFSET	0x1c

#define PINS_PER_BANK		32
#define MUX_PINS_PER_REG	8
#define MUX_PINS_BITS		4
#define MUX_PINS_MASK		0x0f
#define DATA_PINS_PER_REG	32
#define DATA_PINS_BITS		1
#define DATA_PINS_MASK		0x01
#define DLEVEL_PINS_PER_REG	16
#define DLEVEL_PINS_BITS	2
#define DLEVEL_PINS_MASK	0x03
#define PULL_PINS_PER_REG	16
#define PULL_PINS_BITS		2
#define PULL_PINS_MASK		0x03

#define SUNXI_IRQ_NUMBER	32

#define IRQ_CFG_REG		0x200
#define IRQ_CFG_IRQ_PER_REG		8
#define IRQ_CFG_IRQ_BITS		4
#define IRQ_CFG_IRQ_MASK		((1 << IRQ_CFG_IRQ_BITS) - 1)
#define IRQ_CTRL_REG		0x210
#define IRQ_CTRL_IRQ_PER_REG		32
#define IRQ_CTRL_IRQ_BITS		1
#define IRQ_CTRL_IRQ_MASK		((1 << IRQ_CTRL_IRQ_BITS) - 1)
#define IRQ_STATUS_REG		0x214
#define IRQ_STATUS_IRQ_PER_REG		32
#define IRQ_STATUS_IRQ_BITS		1
#define IRQ_STATUS_IRQ_MASK		((1 << IRQ_STATUS_IRQ_BITS) - 1)

#define IRQ_EDGE_RISING		0x00
#define IRQ_EDGE_FALLING	0x01
#define IRQ_LEVEL_HIGH		0x02
#define IRQ_LEVEL_LOW		0x03
#define IRQ_EDGE_BOTH		0x04

struct sunxi_desc_function {
	const char	*name;
	u8		muxval;
	u8		irqnum;
};

struct sunxi_desc_pin {
	struct pinctrl_pin_desc		pin;
	struct sunxi_desc_function	*functions;
};

struct sunxi_pinctrl_desc {
	const struct sunxi_desc_pin	*pins;
	int				npins;
	struct pinctrl_gpio_range	*ranges;
	int				nranges;
};

struct sunxi_pinctrl_function {
	const char	*name;
	const char	**groups;
	unsigned	ngroups;
};

struct sunxi_pinctrl_group {
	const char	*name;
	unsigned long	config;
	unsigned	pin;
};

struct sunxi_pinctrl {
	void __iomem			*membase;
	struct gpio_chip		*chip;
	struct sunxi_pinctrl_desc	*desc;
	struct device			*dev;
#if 0
	struct irq_domain		*domain;
#else
	unsigned int			irq_start;
	struct vmm_host_irq_chip 	*irq_chip;
#endif
	struct sunxi_pinctrl_function	*functions;
	unsigned			nfunctions;
	struct sunxi_pinctrl_group	*groups;
	unsigned			ngroups;
	int				irq;
	int				irq_array[SUNXI_IRQ_NUMBER];
	spinlock_t			lock;
	struct pinctrl_dev		*pctl_dev;
};

#define SUNXI_PIN(_pin, ...)					\
	{							\
		.pin = _pin,					\
		.functions = (struct sunxi_desc_function[]){	\
			__VA_ARGS__, { } },			\
	}

#define SUNXI_FUNCTION(_val, _name)				\
	{							\
		.name = _name,					\
		.muxval = _val,					\
	}

#define SUNXI_FUNCTION_IRQ(_val, _irq)				\
	{							\
		.name = "irq",					\
		.muxval = _val,					\
		.irqnum = _irq,					\
	}

/*
 * The sunXi PIO registers are organized as is:
 * 0x00 - 0x0c	Muxing values.
 *		8 pins per register, each pin having a 4bits value
 * 0x10		Pin values
 *		32 bits per register, each pin corresponding to one bit
 * 0x14 - 0x18	Drive level
 *		16 pins per register, each pin having a 2bits value
 * 0x1c - 0x20	Pull-Up values
 *		16 pins per register, each pin having a 2bits value
 *
 * This is for the first bank. Each bank will have the same layout,
 * with an offset being a multiple of 0x24.
 *
 * The following functions calculate from the pin number the register
 * and the bit offset that we should access.
 */
static inline u32 sunxi_mux_reg(u16 pin)
{
	u8 bank = pin / PINS_PER_BANK;
	u32 offset = bank * BANK_MEM_SIZE;
	offset += MUX_REGS_OFFSET;
	offset += pin % PINS_PER_BANK / MUX_PINS_PER_REG * 0x04;
	return round_down(offset, 4);
}

static inline u32 sunxi_mux_offset(u16 pin)
{
	u32 pin_num = pin % MUX_PINS_PER_REG;
	return pin_num * MUX_PINS_BITS;
}

static inline u32 sunxi_data_reg(u16 pin)
{
	u8 bank = pin / PINS_PER_BANK;
	u32 offset = bank * BANK_MEM_SIZE;
	offset += DATA_REGS_OFFSET;
	offset += pin % PINS_PER_BANK / DATA_PINS_PER_REG * 0x04;
	return round_down(offset, 4);
}

static inline u32 sunxi_data_offset(u16 pin)
{
	u32 pin_num = pin % DATA_PINS_PER_REG;
	return pin_num * DATA_PINS_BITS;
}

static inline u32 sunxi_dlevel_reg(u16 pin)
{
	u8 bank = pin / PINS_PER_BANK;
	u32 offset = bank * BANK_MEM_SIZE;
	offset += DLEVEL_REGS_OFFSET;
	offset += pin % PINS_PER_BANK / DLEVEL_PINS_PER_REG * 0x04;
	return round_down(offset, 4);
}

static inline u32 sunxi_dlevel_offset(u16 pin)
{
	u32 pin_num = pin % DLEVEL_PINS_PER_REG;
	return pin_num * DLEVEL_PINS_BITS;
}

static inline u32 sunxi_pull_reg(u16 pin)
{
	u8 bank = pin / PINS_PER_BANK;
	u32 offset = bank * BANK_MEM_SIZE;
	offset += PULL_REGS_OFFSET;
	offset += pin % PINS_PER_BANK / PULL_PINS_PER_REG * 0x04;
	return round_down(offset, 4);
}

static inline u32 sunxi_pull_offset(u16 pin)
{
	u32 pin_num = pin % PULL_PINS_PER_REG;
	return pin_num * PULL_PINS_BITS;
}

static inline u32 sunxi_irq_cfg_reg(u16 irq)
{
	u8 reg = irq / IRQ_CFG_IRQ_PER_REG;
	return reg + IRQ_CFG_REG;
}

static inline u32 sunxi_irq_cfg_offset(u16 irq)
{
	u32 irq_num = irq % IRQ_CFG_IRQ_PER_REG;
	return irq_num * IRQ_CFG_IRQ_BITS;
}

static inline u32 sunxi_irq_ctrl_reg(u16 irq)
{
	u8 reg = irq / IRQ_CTRL_IRQ_PER_REG;
	return reg + IRQ_CTRL_REG;
}

static inline u32 sunxi_irq_ctrl_offset(u16 irq)
{
	u32 irq_num = irq % IRQ_CTRL_IRQ_PER_REG;
	return irq_num * IRQ_CTRL_IRQ_BITS;
}

static inline u32 sunxi_irq_status_reg(u16 irq)
{
	u8 reg = irq / IRQ_STATUS_IRQ_PER_REG;
	return reg + IRQ_STATUS_REG;
}

static inline u32 sunxi_irq_status_offset(u16 irq)
{
	u32 irq_num = irq % IRQ_STATUS_IRQ_PER_REG;
	return irq_num * IRQ_STATUS_IRQ_BITS;
}

#endif /* __PINCTRL_SUNXI_H */
