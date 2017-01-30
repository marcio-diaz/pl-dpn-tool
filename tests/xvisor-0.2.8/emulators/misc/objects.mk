#/**
# Copyright (c) 2012 Sukanto Ghosh.
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2, or (at your option)
# any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
#
# @file objects.mk
# @author Sukanto Ghosh (sukantoghosh@gmail.com)
# @brief list of misc emulator objects
# */

emulators-objs-$(CONFIG_EMU_MISC_ZERO)+= misc/zero.o
emulators-objs-$(CONFIG_EMU_MISC_A9MPCORE)+= misc/a9mpcore.o
emulators-objs-$(CONFIG_EMU_MISC_ARM11MPCORE)+= misc/arm11mpcore.o
emulators-objs-$(CONFIG_EMU_MISC_PSM)+= misc/xpsm.o
emulators-objs-$(CONFIG_EMU_MISC_FW_CFG)+= misc/fw_cfg.o
emulators-objs-$(CONFIG_EMU_MISC_IMX6_ANATOP)+= misc/imx_anatop.o
emulators-objs-$(CONFIG_EMU_MISC_IMX6_CCM)+= misc/imx_ccm.o
emulators-objs-$(CONFIG_EMU_MISC_IMX6_APBH)+= misc/imx_apbh.o
