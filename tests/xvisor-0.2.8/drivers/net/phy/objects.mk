#/**
# Copyright (c) 2014 Pranav Sawargaonkar.
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
# @author Pranav Sawargaonkar (pranav.sawargaonkar@gmail.com)
# @brief list of driver objects
# */

drivers-objs-$(CONFIG_PHYLIB)+= net/phy/libphy.o
drivers-objs-$(CONFIG_MDIO_SUN4I)+= net/phy/mdio-sun4i.o
drivers-objs-$(CONFIG_MICREL_PHY)+= net/phy/micrel.o

libphy-y += phy.o
libphy-y += mdio_bus.o
libphy-y += phy_device.o
libphy-y += of_mdio.o

%/libphy.o: $(foreach obj,$(libphy-y),%/$(obj))
	$(call merge_objs,$@,$^)

%/libphy.dep: $(foreach dep,$(libphy-y:.o=.dep),%/$(dep))
	$(call merge_deps,$@,$^)
