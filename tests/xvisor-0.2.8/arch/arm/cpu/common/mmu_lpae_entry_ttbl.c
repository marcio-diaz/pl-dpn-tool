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
 * @file mmu_lpae_entry_ttbl.c
 * @author Anup Patel (anup@brainfault.org)
 * @author Jean-Christophe Dubois (jcd@tribudubois.net)
 * @brief Initial translation table setup at reset time
 */

#include <vmm_types.h>
#include <cpu_defines.h>
#include <cpu_mmu_lpae.h>
#include <mmu_lpae.h>

struct mmu_lpae_entry_ctrl {
	u32 ttbl_count;
	int *ttbl_tree;
	u64 *next_ttbl;
	virtual_addr_t ttbl_base;
};

extern u8 def_ttbl[];
extern int def_ttbl_tree[];
#ifdef CONFIG_DEFTERM_EARLY_PRINT
extern u8 defterm_early_base[];
#endif

void __attribute__ ((section(".entry")))
    __setup_initial_ttbl(struct mmu_lpae_entry_ctrl *lpae_entry,
		     virtual_addr_t map_start, virtual_addr_t map_end,
		     virtual_addr_t pa_start, u32 aindex, bool writeable)
{
	u32 i, index;
	u64 *ttbl;
	virtual_addr_t page_addr;

	/* align start addresses */
	map_start &= TTBL_L3_MAP_MASK;
	pa_start &= TTBL_L3_MAP_MASK;

	page_addr = map_start;

	while (page_addr < map_end) {

		/* Setup level1 table */
		ttbl = (u64 *) lpae_entry->ttbl_base;
		index = (page_addr & TTBL_L1_INDEX_MASK) >> TTBL_L1_INDEX_SHIFT;
		if (ttbl[index] & TTBL_VALID_MASK) {
			/* Find level2 table */
			ttbl =
			    (u64 *) (unsigned long)(ttbl[index] &
						    TTBL_OUTADDR_MASK);
		} else {
			/* Allocate new level2 table */
			if (lpae_entry->ttbl_count == TTBL_INITIAL_TABLE_COUNT) {
				while (1) ;	/* No initial table available */
			}
			for (i = 0; i < TTBL_TABLE_ENTCNT; i++) {
				lpae_entry->next_ttbl[i] = 0x0ULL;
			}
			lpae_entry->ttbl_tree[lpae_entry->ttbl_count] =
			    ((virtual_addr_t) ttbl -
			     lpae_entry->ttbl_base) >> TTBL_TABLE_SIZE_SHIFT;
			lpae_entry->ttbl_count++;
			ttbl[index] |=
			    (((virtual_addr_t) lpae_entry->next_ttbl) &
			     TTBL_OUTADDR_MASK);
			ttbl[index] |= (TTBL_TABLE_MASK | TTBL_VALID_MASK);
			ttbl = lpae_entry->next_ttbl;
			lpae_entry->next_ttbl += TTBL_TABLE_ENTCNT;
		}

		/* Setup level2 table */
		index = (page_addr & TTBL_L2_INDEX_MASK) >> TTBL_L2_INDEX_SHIFT;
		if (ttbl[index] & TTBL_VALID_MASK) {
			/* Find level3 table */
			ttbl =
			    (u64 *) (unsigned long)(ttbl[index] &
						    TTBL_OUTADDR_MASK);
		} else {
			/* Allocate new level3 table */
			if (lpae_entry->ttbl_count == TTBL_INITIAL_TABLE_COUNT) {
				while (1) ;	/* No initial table available */
			}
			for (i = 0; i < TTBL_TABLE_ENTCNT; i++) {
				lpae_entry->next_ttbl[i] = 0x0ULL;
			}
			lpae_entry->ttbl_tree[lpae_entry->ttbl_count] =
			    ((virtual_addr_t) ttbl -
			     lpae_entry->ttbl_base) >> TTBL_TABLE_SIZE_SHIFT;
			lpae_entry->ttbl_count++;
			ttbl[index] |=
			    (((virtual_addr_t) lpae_entry->next_ttbl) &
			     TTBL_OUTADDR_MASK);
			ttbl[index] |= (TTBL_TABLE_MASK | TTBL_VALID_MASK);
			ttbl = lpae_entry->next_ttbl;
			lpae_entry->next_ttbl += TTBL_TABLE_ENTCNT;
		}

		/* Setup level3 table */
		index = (page_addr & TTBL_L3_INDEX_MASK) >> TTBL_L3_INDEX_SHIFT;
		if (!(ttbl[index] & TTBL_VALID_MASK)) {
			/* Update level3 table */
			ttbl[index] =
			    (((page_addr - map_start) + pa_start) &
			     TTBL_OUTADDR_MASK);
			ttbl[index] |= TTBL_STAGE1_LOWER_AF_MASK;
			ttbl[index] |= (writeable) ?
			    (TTBL_AP_SRW_U << TTBL_STAGE1_LOWER_AP_SHIFT) :
			    (TTBL_AP_SR_U << TTBL_STAGE1_LOWER_AP_SHIFT);
			ttbl[index] |=
			    (aindex << TTBL_STAGE1_LOWER_AINDEX_SHIFT)
			    & TTBL_STAGE1_LOWER_AINDEX_MASK;
			ttbl[index] |= TTBL_STAGE1_LOWER_NS_MASK;
			ttbl[index] |= (TTBL_SH_INNER_SHAREABLE
					<< TTBL_STAGE1_LOWER_SH_SHIFT);
			ttbl[index] |= (TTBL_TABLE_MASK | TTBL_VALID_MASK);
		}

		/* Point to next page */
		page_addr += TTBL_L3_BLOCK_SIZE;
	}
}

/* Note: This functions must be called with MMU disabled from
 * primary CPU only.
 * Note: This functions cannot refer to any global variable &
 * functions to ensure that it can execute from anywhere.
 */
#define to_load_pa(va)	({ \
			virtual_addr_t _tva = (va); \
			if (exec_start <= _tva && _tva < exec_end) { \
				_tva = _tva - exec_start + load_start; \
			} \
			_tva; \
			})
#define to_exec_va(va)	({ \
			virtual_addr_t _tva = (va); \
			if (load_start <= _tva && _tva < load_end) { \
				_tva = _tva - load_start + exec_start; \
			} \
			_tva; \
			})

#define SECTION_START(SECTION)	_ ## SECTION ## _start
#define SECTION_END(SECTION)	_ ## SECTION ## _end

#define SECTION_ADDR_START(SECTION)	(virtual_addr_t)&SECTION_START(SECTION)
#define SECTION_ADDR_END(SECTION)	(virtual_addr_t)&SECTION_END(SECTION)

#define DECLARE_SECTION(SECTION)					\
	extern virtual_addr_t SECTION_START(SECTION);			\
	extern virtual_addr_t SECTION_END(SECTION)

DECLARE_SECTION(text);
DECLARE_SECTION(cpuinit);
DECLARE_SECTION(spinlock);
DECLARE_SECTION(init);
DECLARE_SECTION(rodata);

#define SETUP_RO_SECTION(ENTRY, SECTION)				\
	__setup_initial_ttbl(&(ENTRY),					\
			     to_exec_va(SECTION_ADDR_START(SECTION)),	\
			     to_exec_va(SECTION_ADDR_END(SECTION)),	\
			     to_load_pa(SECTION_ADDR_START(SECTION)),	\
			     AINDEX_NORMAL_WB,				\
			     FALSE)

void __attribute__ ((section(".entry")))
    _setup_initial_ttbl(virtual_addr_t load_start, virtual_addr_t load_end,
			virtual_addr_t exec_start, virtual_addr_t exec_end)
{
	u32 i;
#ifdef CONFIG_DEFTERM_EARLY_PRINT
	virtual_addr_t defterm_early_va;
#endif
	struct mmu_lpae_entry_ctrl lpae_entry = { 0, NULL, NULL, 0 };

	/* Init ttbl_base, ttbl_tree, and next_ttbl */
	lpae_entry.ttbl_tree =
		(int *)to_load_pa((virtual_addr_t)&def_ttbl_tree);

	for (i = 0; i < TTBL_INITIAL_TABLE_COUNT; i++) {
		lpae_entry.ttbl_tree[i] = -1;
	}

	lpae_entry.ttbl_base = to_load_pa((virtual_addr_t)&def_ttbl);
	lpae_entry.next_ttbl = (u64 *)lpae_entry.ttbl_base;

	/* Init first ttbl */
	for (i = 0; i < TTBL_TABLE_ENTCNT; i++) {
		lpae_entry.next_ttbl[i] = 0x0ULL;
	}

	lpae_entry.ttbl_count++;
	lpae_entry.next_ttbl += TTBL_TABLE_ENTCNT;

#ifdef CONFIG_DEFTERM_EARLY_PRINT
	/* Map UART for early defterm
	 * Note: This is for early debug purpose
	 */
	defterm_early_va = to_exec_va((virtual_addr_t)&defterm_early_base);
	__setup_initial_ttbl(&lpae_entry,
			     defterm_early_va,
			     defterm_early_va + TTBL_L3_BLOCK_SIZE,
			     (virtual_addr_t)CONFIG_DEFTERM_EARLY_BASE_PA,
			     AINDEX_SO, TRUE);
#endif

	/* Map physical = logical
	 * Note: This mapping is using at boot time only
	 */
	__setup_initial_ttbl(&lpae_entry, load_start, load_end, load_start,
			     AINDEX_NORMAL_WB, TRUE);

	/* Map to logical addresses which are
	 * covered by read-only linker sections
	 * Note: This mapping is used at runtime
	 */
	SETUP_RO_SECTION(lpae_entry, text);
	SETUP_RO_SECTION(lpae_entry, init);
	SETUP_RO_SECTION(lpae_entry, cpuinit);
	SETUP_RO_SECTION(lpae_entry, spinlock);
	SETUP_RO_SECTION(lpae_entry, rodata);

	/* Map rest of logical addresses which are
	 * not covered by read-only linker sections
	 * Note: This mapping is used at runtime
	 */
	__setup_initial_ttbl(&lpae_entry, exec_start, exec_end, load_start,
			     AINDEX_NORMAL_WB, TRUE);
}
