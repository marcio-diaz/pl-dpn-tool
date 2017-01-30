/**
 * Copyright (C) 2008 yajin <yajin@vm-kernel.org>
 * Copyright (C) 2010 Sukanto Ghosh.
 * All rights reserved.
 *
 * Modified the qemu bb_nandflash_ecc.c file to remove qemu dependency
 *  - Sukanto
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
 * @file bb_nandflash_ecc.c
 * @author yajin <yajin@vm-kernel.org>
 * @author Sukanto Ghosh <sukantoghosh@gmail.com>
 * @brief Tool to calculate ecc code for beagle nand flash
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <fcntl.h>

typedef u_int8_t uint8_t;
typedef u_int16_t uint16_t;
typedef u_int32_t uint32_t;

#define BB_NAND_PAGE_SIZE   2048
#define BB_NAND_OOB_SIZE     64
#define BB_NAND_SIZE 0x10000000	/*does not include oob */
#define BB_NAND_ECC_OFFSET   0x28

/*
 * Pre-calculated 256-way 1 byte column parity
 */
static const u_char nand_ecc_precalc_table[] = {
	0x00, 0x55, 0x56, 0x03, 0x59, 0x0c, 0x0f, 0x5a,
	0x5a, 0x0f, 0x0c, 0x59, 0x03, 0x56, 0x55, 0x00,
	0x65, 0x30, 0x33, 0x66, 0x3c, 0x69, 0x6a, 0x3f,
	0x3f, 0x6a, 0x69, 0x3c, 0x66, 0x33, 0x30, 0x65,
	0x66, 0x33, 0x30, 0x65, 0x3f, 0x6a, 0x69, 0x3c,
	0x3c, 0x69, 0x6a, 0x3f, 0x65, 0x30, 0x33, 0x66,
	0x03, 0x56, 0x55, 0x00, 0x5a, 0x0f, 0x0c, 0x59,
	0x59, 0x0c, 0x0f, 0x5a, 0x00, 0x55, 0x56, 0x03,
	0x69, 0x3c, 0x3f, 0x6a, 0x30, 0x65, 0x66, 0x33,
	0x33, 0x66, 0x65, 0x30, 0x6a, 0x3f, 0x3c, 0x69,
	0x0c, 0x59, 0x5a, 0x0f, 0x55, 0x00, 0x03, 0x56,
	0x56, 0x03, 0x00, 0x55, 0x0f, 0x5a, 0x59, 0x0c,
	0x0f, 0x5a, 0x59, 0x0c, 0x56, 0x03, 0x00, 0x55,
	0x55, 0x00, 0x03, 0x56, 0x0c, 0x59, 0x5a, 0x0f,
	0x6a, 0x3f, 0x3c, 0x69, 0x33, 0x66, 0x65, 0x30,
	0x30, 0x65, 0x66, 0x33, 0x69, 0x3c, 0x3f, 0x6a,
	0x6a, 0x3f, 0x3c, 0x69, 0x33, 0x66, 0x65, 0x30,
	0x30, 0x65, 0x66, 0x33, 0x69, 0x3c, 0x3f, 0x6a,
	0x0f, 0x5a, 0x59, 0x0c, 0x56, 0x03, 0x00, 0x55,
	0x55, 0x00, 0x03, 0x56, 0x0c, 0x59, 0x5a, 0x0f,
	0x0c, 0x59, 0x5a, 0x0f, 0x55, 0x00, 0x03, 0x56,
	0x56, 0x03, 0x00, 0x55, 0x0f, 0x5a, 0x59, 0x0c,
	0x69, 0x3c, 0x3f, 0x6a, 0x30, 0x65, 0x66, 0x33,
	0x33, 0x66, 0x65, 0x30, 0x6a, 0x3f, 0x3c, 0x69,
	0x03, 0x56, 0x55, 0x00, 0x5a, 0x0f, 0x0c, 0x59,
	0x59, 0x0c, 0x0f, 0x5a, 0x00, 0x55, 0x56, 0x03,
	0x66, 0x33, 0x30, 0x65, 0x3f, 0x6a, 0x69, 0x3c,
	0x3c, 0x69, 0x6a, 0x3f, 0x65, 0x30, 0x33, 0x66,
	0x65, 0x30, 0x33, 0x66, 0x3c, 0x69, 0x6a, 0x3f,
	0x3f, 0x6a, 0x69, 0x3c, 0x66, 0x33, 0x30, 0x65,
	0x00, 0x55, 0x56, 0x03, 0x59, 0x0c, 0x0f, 0x5a,
	0x5a, 0x0f, 0x0c, 0x59, 0x03, 0x56, 0x55, 0x00
};

/**
 * nand_calculate_ecc - [NAND Interface] Calculate 3-byte ECC for 256-byte block
 * @dat:	raw data
 * @ecc_code:	buffer for ECC
 */
int nand_calculate_ecc(const u_char * dat, u_char * ecc_code)
{
	uint8_t idx, reg1, reg2, reg3, tmp1, tmp2;
	int i;

	/* Initialize variables */
	reg1 = reg2 = reg3 = 0;

	/* Build up column parity */
	for (i = 0; i < 256; i++) {
		/* Get CP0 - CP5 from table */
		idx = nand_ecc_precalc_table[*dat++];
		reg1 ^= (idx & 0x3f);

		/* All bit XOR = 1 ? */
		if (idx & 0x40) {
			reg3 ^= (uint8_t) i;
			reg2 ^= ~((uint8_t) i);
		}
	}

	/* Create non-inverted ECC code from line parity */
	tmp1 = (reg3 & 0x80) >> 0;	/* B7 -> B7 */
	tmp1 |= (reg2 & 0x80) >> 1;	/* B7 -> B6 */
	tmp1 |= (reg3 & 0x40) >> 1;	/* B6 -> B5 */
	tmp1 |= (reg2 & 0x40) >> 2;	/* B6 -> B4 */
	tmp1 |= (reg3 & 0x20) >> 2;	/* B5 -> B3 */
	tmp1 |= (reg2 & 0x20) >> 3;	/* B5 -> B2 */
	tmp1 |= (reg3 & 0x10) >> 3;	/* B4 -> B1 */
	tmp1 |= (reg2 & 0x10) >> 4;	/* B4 -> B0 */

	tmp2 = (reg3 & 0x08) << 4;	/* B3 -> B7 */
	tmp2 |= (reg2 & 0x08) << 3;	/* B3 -> B6 */
	tmp2 |= (reg3 & 0x04) << 3;	/* B2 -> B5 */
	tmp2 |= (reg2 & 0x04) << 2;	/* B2 -> B4 */
	tmp2 |= (reg3 & 0x02) << 2;	/* B1 -> B3 */
	tmp2 |= (reg2 & 0x02) << 1;	/* B1 -> B2 */
	tmp2 |= (reg3 & 0x01) << 1;	/* B0 -> B1 */
	tmp2 |= (reg2 & 0x01) << 0;	/* B7 -> B0 */

	/* Calculate final ECC code */
#ifdef CONFIG_MTD_NAND_ECC_SMC
	ecc_code[0] = ~tmp2;
	ecc_code[1] = ~tmp1;
#else
	ecc_code[0] = ~tmp1;
	ecc_code[1] = ~tmp2;
#endif
	ecc_code[2] = ((~reg1) << 2) | 0x03;

	return 0;
}

/*
 *  usage: bb-nandflash-ecc    start_address  size
 */
void useage()
{
	printf("Useage:\n");
	printf("bb_nandflash_ecc nand_img start_address  size\n");
}

/*start_address/size does not include oob
  */
int main(int argc, char **argv)
{
	int retcode = 1;
	uint32_t start_address, size;
	char *nand_image;

	uint32_t pagenumber, pages;

	int nand_fd;
	uint32_t i, j;
	
	uint8_t page_data[BB_NAND_PAGE_SIZE + BB_NAND_OOB_SIZE];
	uint8_t ecc_data[3];
	

	if (argc != 4) {
		useage();
		exit(retcode);
	}

	nand_image = argv[1];

	start_address = strtol(argv[2], NULL, 0);
	size = strtol(argv[3], NULL, 0);

	nand_fd = open(nand_image, O_RDWR);
	if (nand_fd < 0) {
		printf("Can not open nand image %s \n", nand_image);
		exit(retcode);
	}

	if (start_address >= BB_NAND_SIZE) {
		printf("start_address can no be more than 0x%x \n",
		       BB_NAND_SIZE);
		goto error;
	}
	if ((start_address % BB_NAND_PAGE_SIZE) != 0) {
		printf("start_address should be aligned to page boundary \n");
		goto error;
	}

	if (size == 0) {
		printf("size can no be zero \n");
		goto error;
	}
	if ((size % BB_NAND_PAGE_SIZE) != 0) {
		printf("size should be aligned to page boundary \n");
		goto error;
	}

	pagenumber = start_address / BB_NAND_PAGE_SIZE;
	pages = size / BB_NAND_PAGE_SIZE;

	for (i = 0; i < pages; i++) {
		if (lseek
		    (nand_fd,
		     pagenumber * (BB_NAND_PAGE_SIZE + BB_NAND_OOB_SIZE),
		     SEEK_SET) == -1) {
			printf("failed to seek page %d\n", pagenumber + i);
			goto error;
		}
		if (read
		    (nand_fd, page_data,
		     BB_NAND_PAGE_SIZE + BB_NAND_OOB_SIZE) !=
		    (BB_NAND_PAGE_SIZE + BB_NAND_OOB_SIZE)) {
			printf("failed to read page %d\n", pagenumber + i);
			goto error;
		}

		for (j = 0; j < BB_NAND_PAGE_SIZE / 256; j++) {
			nand_calculate_ecc(page_data + j * 256, ecc_data);
			memcpy(page_data + BB_NAND_PAGE_SIZE +
			       BB_NAND_ECC_OFFSET + j * 3, ecc_data, 3);
		}
		if (lseek
		    (nand_fd,
		     pagenumber * (BB_NAND_PAGE_SIZE + BB_NAND_OOB_SIZE),
		     SEEK_SET) == -1) {
			printf("failed to seek page %d\n", pagenumber + i);
			goto error;
		}
		if (write
		    (nand_fd, page_data,
		     BB_NAND_PAGE_SIZE + BB_NAND_OOB_SIZE) !=
		    (BB_NAND_PAGE_SIZE + BB_NAND_OOB_SIZE)) {
			printf("failed to write page %d\n", pagenumber + i);
			goto error;
		}
		pagenumber++;
	}

	retcode = 0;
 error:
	close(nand_fd);
	return (retcode);
}
