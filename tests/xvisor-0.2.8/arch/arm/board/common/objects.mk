#/**
# Copyright (c) 2011 Anup Patel.
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
# @author Anup Patel (anup@brainfault.org)
# @brief list of common objects.
# */

board-common-objs-y+=devtree.o
board-common-objs-y+=defterm.o
board-common-objs-$(CONFIG_DEFTERM_EARLY_PRINT)+=defterm_early.o
board-common-objs-$(CONFIG_ARM_SMP_OPS)+=smp_ops.o
board-common-objs-$(CONFIG_ARM_SMP_IPI)+=smp_ipi.o
board-common-objs-$(CONFIG_ARM_SMP_SPIN_TABLE)+=smp_spin_table.o
board-common-objs-$(CONFIG_ARM_SCU)+=smp_scu.o
board-common-objs-$(CONFIG_ARM_SMP_PSCI)+=smp_psci.o
board-common-objs-$(CONFIG_ARM_SMP_IMX)+=smp_imx.o
