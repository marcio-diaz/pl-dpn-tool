#/**
# Copyright (c) 2012 Anup Patel.
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
# @brief list of driver objects
# */

drivers-objs-$(CONFIG_RTC)+= rtc/rtc_core.o

rtc_core-y += rtc-dev.o
rtc_core-y += rtc-lib.o

%/rtc_core.o: $(foreach obj,$(rtc_core-y),%/$(obj))
	$(call merge_objs,$@,$^)

%/rtc_core.dep: $(foreach dep,$(rtc_core-y:.o=.dep),%/$(dep))
	$(call merge_deps,$@,$^)

drivers-objs-$(CONFIG_RTC_PL031)+= rtc/rtc-pl031.o
drivers-objs-$(CONFIG_RTC_S3C)+= rtc/rtc-s3c.o

