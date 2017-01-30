/**
 * Copyright (c) 2016 Anup Patel.
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
 * @file vsdaemon.c
 * @author Anup Patel (anup@brainfault.org)
 * @brief vserial daemon library implementation
 */

#include <vmm_error.h>
#include <vmm_macros.h>
#include <vmm_heap.h>
#include <vmm_mutex.h>
#include <vmm_modules.h>
#include <libs/stringlib.h>
#include <libs/vsdaemon.h>

#define MODULE_DESC			"vserial daemon library"
#define MODULE_AUTHOR			"Anup Patel"
#define MODULE_LICENSE			"GPL"
#define MODULE_IPRIORITY		VSDAEMON_IPRIORITY
#define	MODULE_INIT			vsdaemon_init
#define	MODULE_EXIT			vsdaemon_exit

struct vsdaemon_control {
	struct vmm_mutex vsd_list_lock;
	struct dlist vsd_list;
	struct dlist vsd_trans_list;
	struct vmm_notifier_block vser_client;
};

static struct vsdaemon_control vsdc;

int vsdaemon_transport_register(struct vsdaemon_transport *trans)
{
	int rc = VMM_OK;
	bool found;
	struct vsdaemon_transport *t;

	BUG_ON(!vmm_scheduler_orphan_context());

	if (!trans) {
		return VMM_EINVALID;
	}

	vmm_mutex_lock(&vsdc.vsd_list_lock);

	found = FALSE;
	list_for_each_entry(t, &vsdc.vsd_trans_list, head) {
		if (!strncmp(t->name, trans->name, sizeof(t->name))) {
			found = TRUE;
			break;
		}
	}
	if (found) {
		rc = VMM_EEXIST;
		goto fail;
	}

	INIT_LIST_HEAD(&trans->head);
	trans->use_count = 0;
	list_add_tail(&trans->head, &vsdc.vsd_trans_list);

fail:
	vmm_mutex_unlock(&vsdc.vsd_list_lock);

	return rc;
}
VMM_EXPORT_SYMBOL(vsdaemon_transport_register);

int vsdaemon_transport_unregister(struct vsdaemon_transport *trans)
{
	int rc = VMM_OK;
	bool found;
	struct vsdaemon_transport *t;

	BUG_ON(!vmm_scheduler_orphan_context());

	if (!trans) {
		return VMM_EINVALID;
	}

	vmm_mutex_lock(&vsdc.vsd_list_lock);

	found = FALSE;
	list_for_each_entry(t, &vsdc.vsd_trans_list, head) {
		if (t == trans) {
			found = TRUE;
			break;
		}
	}
	if (!found) {
		rc = VMM_ENOTAVAIL;
		goto fail;
	}

	if (trans->use_count) {
		rc = VMM_EBUSY;
		goto fail;
	}
	list_del(&trans->head);

fail:
	vmm_mutex_unlock(&vsdc.vsd_list_lock);

	return rc;
}
VMM_EXPORT_SYMBOL(vsdaemon_transport_unregister);

/* Note: Must be called with vsd_list_lock held */
static struct vsdaemon_transport *__vsdaemon_transport_find(const char *trans)
{
	bool found;
	struct vsdaemon_transport *t;

	found = FALSE;
	list_for_each_entry(t, &vsdc.vsd_trans_list, head) {
		if (!strncmp(t->name, trans, sizeof(t->name))) {
			found = TRUE;
			break;
		}
	}

	return (found) ? t : NULL;
}

struct vsdaemon_transport *vsdaemon_transport_get(int index)
{
	bool found;
	struct vsdaemon_transport *trans;

	BUG_ON(!vmm_scheduler_orphan_context());

	if (index < 0) {
		return NULL;
	}

	vmm_mutex_lock(&vsdc.vsd_list_lock);

	trans = NULL;
	found = FALSE;
	list_for_each_entry(trans, &vsdc.vsd_trans_list, head) {
		if (!index) {
			found = TRUE;
			break;
		}
		index--;
	}

	vmm_mutex_unlock(&vsdc.vsd_list_lock);

	if (!found) {
		return NULL;
	}

	return trans;
}
VMM_EXPORT_SYMBOL(vsdaemon_transport_get);

u32 vsdaemon_transport_count(void)
{
	u32 retval = 0;
	struct vsdaemon_transport *trans;

	BUG_ON(!vmm_scheduler_orphan_context());

	vmm_mutex_lock(&vsdc.vsd_list_lock);

	list_for_each_entry(trans, &vsdc.vsd_trans_list, head) {
		retval++;
	}

	vmm_mutex_unlock(&vsdc.vsd_list_lock);

	return retval;
}
VMM_EXPORT_SYMBOL(vsdaemon_transport_count);

static void vsdaemon_vserial_recv(struct vmm_vserial *vser, void *priv, u8 ch)
{
	struct vsdaemon *vsd = priv;

	vsd->trans->receive_char(vsd, ch);
}

static int vsdaemon_main(void *data)
{
	struct vsdaemon *vsd = data;

	return vsd->trans->main_loop(vsd);
}

int vsdaemon_create(const char *transport_name,
		    const char *vserial_name,
		    const char *daemon_name,
		    int argc, char **argv)
{
	int rc = VMM_OK;
	bool found;
	struct vsdaemon *vsd;
	struct vsdaemon_transport *trans;
	struct vmm_vserial *vser;

	BUG_ON(!vmm_scheduler_orphan_context());

	if (!transport_name || !vserial_name || !daemon_name) {
		return VMM_EINVALID;
	}

	vser = vmm_vserial_find(vserial_name);
	if (!vser) {
		return VMM_EINVALID;
	}

	vmm_mutex_lock(&vsdc.vsd_list_lock);

	trans = __vsdaemon_transport_find(transport_name);
	if (!trans) {
		rc = VMM_EINVALID;
		goto fail1;
	}

	found = FALSE;
	list_for_each_entry(vsd, &vsdc.vsd_list, head) {
		if (!strncmp(vsd->name, daemon_name, sizeof(vsd->name))) {
			found = TRUE;
			break;
		}
	}
	if (found) {
		rc = VMM_EEXIST;
		goto fail1;
	}

	vsd = vmm_zalloc(sizeof(struct vsdaemon));
	if (!vsd) {
		rc = VMM_ENOMEM;
		goto fail1;
	}
	INIT_LIST_HEAD(&vsd->head);
	strlcpy(vsd->name, daemon_name, sizeof(vsd->name) - 1);
	vsd->trans = trans;
	vsd->vser = vser;

	rc = vsd->trans->setup(vsd, argc, argv);
	if (rc) {
		goto fail2;
	}

	rc = vmm_vserial_register_receiver(vser, &vsdaemon_vserial_recv, vsd);
	if (rc) {
		goto fail3;
	}

	vsd->thread = vmm_threads_create(vsd->name, &vsdaemon_main, vsd, 
					 VMM_THREAD_DEF_PRIORITY,
					 VMM_THREAD_DEF_TIME_SLICE);
	if (!vsd->thread) {
		rc = VMM_EFAIL;
		goto fail4;
	}

	list_add_tail(&vsd->head, &vsdc.vsd_list);
	vsd->trans->use_count++;

	vmm_threads_start(vsd->thread);

	vmm_mutex_unlock(&vsdc.vsd_list_lock);

	return VMM_OK;

fail4:
	vmm_vserial_unregister_receiver(vser, &vsdaemon_vserial_recv, vsd);
fail3:
	vsd->trans->cleanup(vsd);
fail2:
	vmm_free(vsd);
fail1:
	vmm_mutex_unlock(&vsdc.vsd_list_lock);
	return rc;
}
VMM_EXPORT_SYMBOL(vsdaemon_create);

/* Note: must be called with vsd_list_lock held */
static int __vsdaemon_destroy(struct vsdaemon *vsd)
{
	vmm_threads_stop(vsd->thread);

	vsd->trans->use_count--;
	list_del(&vsd->head);

	vmm_threads_destroy(vsd->thread);

	vmm_vserial_unregister_receiver(vsd->vser, 
					&vsdaemon_vserial_recv, vsd);

	vsd->trans->cleanup(vsd);

	vmm_free(vsd);

	return VMM_OK;
}

int vsdaemon_destroy(const char *daemon_name)
{
	int rc = VMM_OK;
	bool found;
	struct vsdaemon *vsd;

	if (!daemon_name) {
		return VMM_EINVALID;
	}

	vmm_mutex_lock(&vsdc.vsd_list_lock);

	vsd = NULL;
	found = FALSE;
	list_for_each_entry(vsd, &vsdc.vsd_list, head) {
		if (!strncmp(vsd->name, daemon_name, sizeof(vsd->name))) {
			found = TRUE;
			break;
		}
	}
	if (!found) {
		rc = VMM_EINVALID;
		goto done;
	}

	rc = __vsdaemon_destroy(vsd);

done:
	vmm_mutex_unlock(&vsdc.vsd_list_lock);

	return rc;
}
VMM_EXPORT_SYMBOL(vsdaemon_destroy);

struct vsdaemon *vsdaemon_get(int index)
{
	bool found;
	struct vsdaemon *vsd;

	BUG_ON(!vmm_scheduler_orphan_context());

	if (index < 0) {
		return NULL;
	}

	vmm_mutex_lock(&vsdc.vsd_list_lock);

	vsd = NULL;
	found = FALSE;
	list_for_each_entry(vsd, &vsdc.vsd_list, head) {
		if (!index) {
			found = TRUE;
			break;
		}
		index--;
	}

	vmm_mutex_unlock(&vsdc.vsd_list_lock);

	if (!found) {
		return NULL;
	}

	return vsd;
}
VMM_EXPORT_SYMBOL(vsdaemon_get);

u32 vsdaemon_count(void)
{
	u32 retval = 0;
	struct vsdaemon *vsd;

	BUG_ON(!vmm_scheduler_orphan_context());

	vmm_mutex_lock(&vsdc.vsd_list_lock);

	list_for_each_entry(vsd, &vsdc.vsd_list, head) {
		retval++;
	}

	vmm_mutex_unlock(&vsdc.vsd_list_lock);

	return retval;
}
VMM_EXPORT_SYMBOL(vsdaemon_count);

static int vsdaemon_vserial_notification(struct vmm_notifier_block *nb,
					 unsigned long evt, void *data)
{
	bool found;
	u32 destroy_count = 0;
	struct vsdaemon *vsd;
	struct vmm_vserial_event *e = data;

	if (evt != VMM_VSERIAL_EVENT_DESTROY) {
		/* We are only interested in destroy events so,
		 * don't care about this event.
		 */
		return NOTIFY_DONE;
	}

	vmm_mutex_lock(&vsdc.vsd_list_lock);

again:
	found = FALSE;
	vsd = NULL;
	list_for_each_entry(vsd, &vsdc.vsd_list, head) {
		if (vsd->vser == e->vser) {
			found = TRUE;
			break;
		}
	}

	if (found) {
		__vsdaemon_destroy(vsd);
		destroy_count++;
		goto again;
	}

	vmm_mutex_unlock(&vsdc.vsd_list_lock);

	if (!destroy_count) {
		/* Did not find suitable vsdaemon so,
		 * don't care about this event.
		 */
		return NOTIFY_DONE;
	}

	return NOTIFY_OK;
}

static int __init vsdaemon_init(void)
{
	int rc;

	memset(&vsdc, 0, sizeof(vsdc));

	INIT_MUTEX(&vsdc.vsd_list_lock);
	INIT_LIST_HEAD(&vsdc.vsd_list);
	INIT_LIST_HEAD(&vsdc.vsd_trans_list);

	vsdc.vser_client.notifier_call = &vsdaemon_vserial_notification;
	vsdc.vser_client.priority = 0;
	rc = vmm_vserial_register_client(&vsdc.vser_client);
	if (rc) {
		return rc;
	}

	return VMM_OK;
}

static void __exit vsdaemon_exit(void)
{
	vmm_vserial_unregister_client(&vsdc.vser_client);
}

VMM_DECLARE_MODULE(MODULE_DESC, 
			MODULE_AUTHOR, 
			MODULE_LICENSE, 
			MODULE_IPRIORITY, 
			MODULE_INIT, 
			MODULE_EXIT);
