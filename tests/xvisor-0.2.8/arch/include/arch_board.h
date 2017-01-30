/**
 * Copyright (c) 2012 Anup Patel.
 * All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2, or (at your option)
 * any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 *
 * @file arch_board.h
 * @author Anup Patel (anup@brainfault.org)
 * @brief generic interface for arch specific Board functions
 */
#ifndef _ARCH_BOARD_H__
#define _ARCH_BOARD_H__

#include <vmm_chardev.h>

/** Print board specific information */
void arch_board_print_info(struct vmm_chardev *cdev);

/** Board early init */
int arch_board_early_init(void);

/** Board final init */
int arch_board_final_init(void);

#endif
