#/**
# Copyright (c) 2014 Anup Patel.
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
# @brief list of Sun4i DTBs.
# */

board-dtbs-$(CONFIG_ARMV7A)+=dts/sun4i/cubieboard/one_guest_pb-a8.dtb
board-dtbs-$(CONFIG_ARMV7A)+=dts/sun4i/cubieboard/one_guest_vexpress-a9.dtb
board-dtbs-$(CONFIG_ARMV7A)+=dts/sun4i/cubieboard/two_guest_pb-a8.dtb

board-dtbs-$(CONFIG_ARMV7A)+=dts/sun4i/hackberry/one_guest_pb-a8.dtb
board-dtbs-$(CONFIG_ARMV7A)+=dts/sun4i/hackberry/one_guest_vexpress-a9.dtb
board-dtbs-$(CONFIG_ARMV7A)+=dts/sun4i/hackberry/two_guest_pb-a8.dtb
