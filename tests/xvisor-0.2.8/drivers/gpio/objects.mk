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

drivers-objs-$(CONFIG_GPIOLIB)+= gpio/gpiolib.o
drivers-objs-$(CONFIG_GPIOLIB)+= gpio/gpiolib-legacy.o
drivers-objs-$(CONFIG_GPIO_GENERIC)+= gpio/gpio-generic.o
drivers-objs-$(CONFIG_OF_GPIO)+= gpio/gpiolib-of.o
drivers-objs-$(CONFIG_GPIO_MXC)+= gpio/gpio-mxc.o
