#/**
# Copyright (C) 2015 Institut de Recherche Technologique SystemX and OpenWide.
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
# @author Jimmy Durand Wesolowski <jimmy.durand-wesolowski@openwide.fr>
# @brief Makefile for the Freescale network device drivers.
# */

drivers-objs-$(CONFIG_FEC) += net/ethernet/freescale/fec.o

fec-y += fec_main.o
# fec-y += fec_ptp.o

%/fec.o: $(foreach obj,$(fec-y),%/$(obj))
	$(call merge_objs,$@,$^)
