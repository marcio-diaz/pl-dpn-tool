#ifndef _LINUX_DEVICE_H
#define _LINUX_DEVICE_H

#include <vmm_stdio.h>
#include <vmm_devtree.h>
#include <vmm_devres.h>
#include <vmm_devdrv.h>

#include <linux/err.h>
#include <linux/errno.h>
#include <linux/io.h>
#include <linux/cache.h>
#include <linux/printk.h>
#include <linux/module.h>
#include <linux/of.h>
#include <linux/pm.h>

#define bus_type			vmm_bus
#define device_type			vmm_device_type
#define device				vmm_device
#define device_driver			vmm_driver

#define class_register(cls)		vmm_devdrv_register_class(cls)
#define class_unregister(cls)		vmm_devdrv_unregister_class(cls)

#define bus_register(bus)		vmm_devdrv_register_bus(bus)
#define bus_unregister(bus)		vmm_devdrv_unregister_bus(bus)
#define bus_register_notifier(bus, nb)	vmm_devdrv_bus_register_notifier(bus, nb)
#define bus_unregister_notifier(bus, nb) \
					vmm_devdrv_bus_unregister_notifier(bus, nb)

#define BUS_NOTIFY_ADD_DEVICE		VMM_BUS_NOTIFY_ADD_DEVICE
#define BUS_NOTIFY_DEL_DEVICE		VMM_BUS_NOTIFY_DEL_DEVICE
#define BUS_NOTIFY_BIND_DRIVER		VMM_BUS_NOTIFY_BIND_DRIVER
#define BUS_NOTIFY_BOUND_DRIVER		VMM_BUS_NOTIFY_BOUND_DRIVER
#define BUS_NOTIFY_UNBIND_DRIVER	VMM_BUS_NOTIFY_UNBIND_DRIVER
#define BUS_NOTIFY_UNBOUND_DRIVER	VMM_BUS_NOTIFY_UNBOUND_DRIVER

#define get_device(dev)			vmm_devdrv_ref_device(dev)
#define put_device(dev)			vmm_devdrv_dref_device(dev)
#define device_is_registered(dev)	vmm_devdrv_isregistered_device(dev)
#define dev_name(dev)			(dev)->name
#define dev_set_name(dev, msg...)	vmm_sprintf((dev)->name, msg)
#define device_initialize(dev)		vmm_devdrv_initialize_device(dev)
#define device_add(dev)			vmm_devdrv_register_device(dev)
#define device_attach(dev)		vmm_devdrv_attach_device(dev)
#define device_bind_driver(dev)		vmm_devdrv_attach_device(dev)
#define device_release_driver(dev)	vmm_devdrv_dettach_device(dev)
#define device_del(dev)			vmm_devdrv_unregister_device(dev)
#define device_register(dev)		vmm_devdrv_register_device(dev)
#define device_unregister(dev)		vmm_devdrv_unregister_device(dev)
#define device_lock(_lock)		do { } while (0);
#define device_trylock(_lock)		(1) /* always available */
#define device_unlock(_lock)		do { } while (0);

#define driver_register(drv)		vmm_devdrv_register_driver(drv)
#define driver_attach(drv)		vmm_devdrv_attach_driver(drv)
#define driver_dettach(drv)		vmm_devdrv_dettach_driver(drv)
#define driver_unregister(drv)		vmm_devdrv_unregister_driver(drv)

#define dev_get_drvdata(dev)		vmm_devdrv_get_data(dev)
#define dev_set_drvdata(dev, data)	vmm_devdrv_set_data(dev, data)

#define	platform_set_drvdata(pdev, data) \
				do { (pdev)->priv = (void *)data; } while (0)
#define	platform_get_drvdata(pdev)	(pdev)->priv

#define dr_release_t			vmm_dr_release_t
#define dr_match_t			vmm_dr_match_t

#define devres_alloc(release, size, gfp) \
					vmm_devres_alloc(release, size)
#define devres_for_each_res(dev, release, match, match_data, fn, data) \
					vmm_devres_for_each_res(dev, release, \
						match, match_data, fn, data)
#define devres_free(res)		vmm_devres_free(res)
#define devres_add(dev, res)		vmm_devres_add(dev, res)
#define devres_find(dev, release, match, match_data) \
					vmm_devres_find(dev, release, \
						match, match_data)
#define devres_get(dev, new_res, match, match_data) \
					vmm_devres_get(dev, new_res, \
						match, match_data)
#define devres_remove(dev, release, match, match_data) \
					vmm_devres_remove(dev, release, \
						match, match_data)
#define devres_destroy(dev, release, match, match_data) \
					vmm_devres_destroy(dev, release, \
						match, match_data)
#define devres_release(dev, release, match, match_data) \
					vmm_devres_release(dev, release, \
						match, match_data)
#define devres_release_all(dev)		vmm_devres_release_all(dev)

#define devm_kmalloc(dev, size, gfp)	vmm_devm_malloc(dev, size)
#define devm_kzalloc(dev, size, gfp)	vmm_devm_zalloc(dev, size)
#define devm_kmalloc_array(dev, n, size, flags) \
					vmm_devm_malloc_array(dev, n, size)
#define devm_kcalloc(dev, n, size, flags) \
					vmm_devm_calloc(dev, n, size)
#define devm_kfree(dev, p)		vmm_devm_free(dev, p)
#define devm_kstrdup(dev, s, gfp)	vmm_devm_strdup(dev, s)

#define bus_find_device(bus, start, data, match)	\
			vmm_devdrv_bus_find_device(bus, start, data, match)
#define bus_find_device_by_name(bus, start, name)	\
			vmm_devdrv_bus_find_device_by_name(bus, start, name)
#define bus_for_each_device(bus, start, data, fn)	\
			vmm_devdrv_bus_device_iterate(bus, start, data, fn)
#define bus_for_each_dev(bus, start, data, fn)	\
			vmm_devdrv_bus_device_iterate(bus, start, data, fn)
#define bus_for_each_drv(bus, start, data, fn)	\
			vmm_devdrv_bus_driver_iterate(bus, start, data, fn)

#define device_for_each_child(dev, data, fn)		\
			vmm_devdrv_for_each_child(dev, data, fn)

static inline int dev_to_node(struct device *dev) { return -1; }
static inline void set_dev_node(struct device *dev, int node) { }

#define dev_WARN	dev_warn
#define dev_WARN_ONCE(dev, condition, format, arg...)	\
	dev_warn(dev, format, ##arg)

/**
 * module_driver() - Helper macro for drivers that don't do anything
 * special in module init/exit. This eliminates a lot of boilerplate.
 * Each module may only use this macro once, and calling it replaces
 * module_init() and module_exit().
 *
 * @__driver: driver name
 * @__register: register function for this driver type
 * @__unregister: unregister function for this driver type
 * @...: Additional arguments to be passed to __register and __unregister.
 *
 * Use this macro to construct bus specific macros for registering
 * drivers, and do not use it on its own.
 */
#define module_driver(__desc, __author, __license, __ipriority,	\
		      __driver, __register, __unregister, ...)	\
static int __init __driver##_init(void)				\
{								\
	return __register(&(__driver) , ##__VA_ARGS__);		\
}								\
static void __exit __driver##_exit(void)			\
{								\
	__unregister(&(__driver) , ##__VA_ARGS__);		\
}								\
VMM_DECLARE_MODULE(__desc,					\
		   __author,					\
		   __license,					\
		   __ipriority,					\
		   __driver##_init,				\
		   __driver##_exit);


/* interface for exporting device attributes */

#endif /* _LINUX_DEVICE_H */
