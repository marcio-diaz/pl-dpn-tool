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
 * @file input.h
 * @author Anup Patel (anup@brainfault.org)
 * @brief Input device framework header
 *
 * The source has been largely adapted from Linux 3.x or higher:
 * include/linux/input.h
 *
 * Copyright (c) 1999-2002 Vojtech Pavlik
 *
 * The original code is licensed under the GPL.
 */

#ifndef __DRV_INPUT_H_
#define __DRV_INPUT_H_

#include <vmm_types.h>
#include <vmm_spinlocks.h>
#include <vmm_timer.h>
#include <vmm_devdrv.h>
#include <libs/list.h>
#include <libs/bitmap.h>

/** Input module intialization priority */
#define INPUT_IPRIORITY					1

/** Input device class details */
#define INPUT_DEV_CLASS_NAME				"input"
#define INPUT_DEV_CLASS_IPRIORITY			INPUT_IPRIORITY

/** Input id structure */
struct input_id {
	u16 bustype;
	u16 vendor;
	u16 product;
	u16 version;
};

/**
 * struct input_absinfo -
 * @value: latest reported value for the axis.
 * @minimum: specifies minimum value for the axis.
 * @maximum: specifies maximum value for the axis.
 * @fuzz: specifies fuzz value that is used to filter noise from
 *	the event stream.
 * @flat: values that are within this value will be discarded by
 *	joydev interface and reported as 0 instead.
 * @resolution: specifies resolution for the values reported for
 *	the axis.
 *
 * Note that input core does not clamp reported values to the
 * [minimum, maximum] limits, such task is left to userspace.
 *
 * Resolution for main axes (ABS_X, ABS_Y, ABS_Z) is reported in
 * units per millimeter (units/mm), resolution for rotational axes
 * (ABS_RX, ABS_RY, ABS_RZ) is reported in units per radian.
 */
struct input_absinfo {
	s32 value;
	s32 minimum;
	s32 maximum;
	s32 fuzz;
	s32 flat;
	s32 resolution;
};

/**
 * struct input_keymap_entry - keymap data
 * @scancode: scancode represented in machine-endian form.
 * @len: length of the scancode that resides in @scancode buffer.
 * @index: index in the keymap, may be used instead of scancode
 * @flags: allows to specify how to the request. For example,
 *	setting INPUT_KEYMAP_BY_INDEX flag indicates that kernel
 *	should perform lookup in keymap by @index instead of @scancode
 * @keycode: key code assigned to this scancode
 *
 * The structure is used to retrieve and modify keymap data. Users have
 * option of performing lookup either by @scancode itself or by @index
 * in keymap entry. EVIOCGKEYCODE will also return scancode or index
 * (depending on which element was used to perform lookup).
 */
struct input_keymap_entry {
#define INPUT_KEYMAP_BY_INDEX	(1 << 0)
	u8  flags;
	u8  len;
	u16 index;
	u32 keycode;
	u8  scancode[32];
};

/*
 * Device properties and quirks
 */

#define INPUT_PROP_POINTER		0x00	/* needs a pointer */
#define INPUT_PROP_DIRECT		0x01	/* direct input devices */
#define INPUT_PROP_BUTTONPAD		0x02	/* has button(s) under pad */
#define INPUT_PROP_SEMI_MT		0x03	/* touch rectangle only */

#define INPUT_PROP_MAX			0x1f
#define INPUT_PROP_CNT			(INPUT_PROP_MAX + 1)

/*
 * Event types
 */

#define EV_SYN			0x00
#define EV_KEY			0x01
#define EV_REL			0x02
#define EV_ABS			0x03
#define EV_MSC			0x04
#define EV_SW			0x05
#define EV_LED			0x11
#define EV_SND			0x12
#define EV_REP			0x14
#define EV_FF			0x15
#define EV_PWR			0x16
#define EV_FF_STATUS		0x17
#define EV_MAX			0x1f
#define EV_CNT			(EV_MAX+1)

/*
 * Synchronization events.
 */

#define SYN_REPORT		0
#define SYN_CONFIG		1
#define SYN_MT_REPORT		2
#define SYN_DROPPED		3

/*
 * Keys and buttons
 *
 * Most of the keys/buttons are modeled after USB HUT 1.12
 * (see http://www.usb.org/developers/hidpage).
 * Abbreviations in the comments:
 * AC - Application Control
 * AL - Application Launch Button
 * SC - System Control
 */

#define KEY_RESERVED		0
#define KEY_ESC			1
#define KEY_1			2
#define KEY_2			3
#define KEY_3			4
#define KEY_4			5
#define KEY_5			6
#define KEY_6			7
#define KEY_7			8
#define KEY_8			9
#define KEY_9			10
#define KEY_0			11
#define KEY_MINUS		12
#define KEY_EQUAL		13
#define KEY_BACKSPACE		14
#define KEY_TAB			15
#define KEY_Q			16
#define KEY_W			17
#define KEY_E			18
#define KEY_R			19
#define KEY_T			20
#define KEY_Y			21
#define KEY_U			22
#define KEY_I			23
#define KEY_O			24
#define KEY_P			25
#define KEY_LEFTBRACE		26
#define KEY_RIGHTBRACE		27
#define KEY_ENTER		28
#define KEY_LEFTCTRL		29
#define KEY_A			30
#define KEY_S			31
#define KEY_D			32
#define KEY_F			33
#define KEY_G			34
#define KEY_H			35
#define KEY_J			36
#define KEY_K			37
#define KEY_L			38
#define KEY_SEMICOLON		39
#define KEY_APOSTROPHE		40
#define KEY_GRAVE		41
#define KEY_LEFTSHIFT		42
#define KEY_BACKSLASH		43
#define KEY_Z			44
#define KEY_X			45
#define KEY_C			46
#define KEY_V			47
#define KEY_B			48
#define KEY_N			49
#define KEY_M			50
#define KEY_COMMA		51
#define KEY_DOT			52
#define KEY_SLASH		53
#define KEY_RIGHTSHIFT		54
#define KEY_KPASTERISK		55
#define KEY_LEFTALT		56
#define KEY_SPACE		57
#define KEY_CAPSLOCK		58
#define KEY_F1			59
#define KEY_F2			60
#define KEY_F3			61
#define KEY_F4			62
#define KEY_F5			63
#define KEY_F6			64
#define KEY_F7			65
#define KEY_F8			66
#define KEY_F9			67
#define KEY_F10			68
#define KEY_NUMLOCK		69
#define KEY_SCROLLLOCK		70
#define KEY_KP7			71
#define KEY_KP8			72
#define KEY_KP9			73
#define KEY_KPMINUS		74
#define KEY_KP4			75
#define KEY_KP5			76
#define KEY_KP6			77
#define KEY_KPPLUS		78
#define KEY_KP1			79
#define KEY_KP2			80
#define KEY_KP3			81
#define KEY_KP0			82
#define KEY_KPDOT		83

#define KEY_ZENKAKUHANKAKU	85
#define KEY_102ND		86
#define KEY_F11			87
#define KEY_F12			88
#define KEY_RO			89
#define KEY_KATAKANA		90
#define KEY_HIRAGANA		91
#define KEY_HENKAN		92
#define KEY_KATAKANAHIRAGANA	93
#define KEY_MUHENKAN		94
#define KEY_KPJPCOMMA		95
#define KEY_KPENTER		96
#define KEY_RIGHTCTRL		97
#define KEY_KPSLASH		98
#define KEY_SYSRQ		99
#define KEY_RIGHTALT		100
#define KEY_LINEFEED		101
#define KEY_HOME		102
#define KEY_UP			103
#define KEY_PAGEUP		104
#define KEY_LEFT		105
#define KEY_RIGHT		106
#define KEY_END			107
#define KEY_DOWN		108
#define KEY_PAGEDOWN		109
#define KEY_INSERT		110
#define KEY_DELETE		111
#define KEY_MACRO		112
#define KEY_MUTE		113
#define KEY_VOLUMEDOWN		114
#define KEY_VOLUMEUP		115
#define KEY_POWER		116	/* SC System Power Down */
#define KEY_KPEQUAL		117
#define KEY_KPPLUSMINUS		118
#define KEY_PAUSE		119
#define KEY_SCALE		120	/* AL Compiz Scale (Expose) */

#define KEY_KPCOMMA		121
#define KEY_HANGEUL		122
#define KEY_HANGUEL		KEY_HANGEUL
#define KEY_HANJA		123
#define KEY_YEN			124
#define KEY_LEFTMETA		125
#define KEY_RIGHTMETA		126
#define KEY_COMPOSE		127

#define KEY_STOP		128	/* AC Stop */
#define KEY_AGAIN		129
#define KEY_PROPS		130	/* AC Properties */
#define KEY_UNDO		131	/* AC Undo */
#define KEY_FRONT		132
#define KEY_COPY		133	/* AC Copy */
#define KEY_OPEN		134	/* AC Open */
#define KEY_PASTE		135	/* AC Paste */
#define KEY_FIND		136	/* AC Search */
#define KEY_CUT			137	/* AC Cut */
#define KEY_HELP		138	/* AL Integrated Help Center */
#define KEY_MENU		139	/* Menu (show menu) */
#define KEY_CALC		140	/* AL Calculator */
#define KEY_SETUP		141
#define KEY_SLEEP		142	/* SC System Sleep */
#define KEY_WAKEUP		143	/* System Wake Up */
#define KEY_FILE		144	/* AL Local Machine Browser */
#define KEY_SENDFILE		145
#define KEY_DELETEFILE		146
#define KEY_XFER		147
#define KEY_PROG1		148
#define KEY_PROG2		149
#define KEY_WWW			150	/* AL Internet Browser */
#define KEY_MSDOS		151
#define KEY_COFFEE		152	/* AL Terminal Lock/Screensaver */
#define KEY_SCREENLOCK		KEY_COFFEE
#define KEY_DIRECTION		153
#define KEY_CYCLEWINDOWS	154
#define KEY_MAIL		155
#define KEY_BOOKMARKS		156	/* AC Bookmarks */
#define KEY_COMPUTER		157
#define KEY_BACK		158	/* AC Back */
#define KEY_FORWARD		159	/* AC Forward */
#define KEY_CLOSECD		160
#define KEY_EJECTCD		161
#define KEY_EJECTCLOSECD	162
#define KEY_NEXTSONG		163
#define KEY_PLAYPAUSE		164
#define KEY_PREVIOUSSONG	165
#define KEY_STOPCD		166
#define KEY_RECORD		167
#define KEY_REWIND		168
#define KEY_PHONE		169	/* Media Select Telephone */
#define KEY_ISO			170
#define KEY_CONFIG		171	/* AL Consumer Control Configuration */
#define KEY_HOMEPAGE		172	/* AC Home */
#define KEY_REFRESH		173	/* AC Refresh */
#define KEY_EXIT		174	/* AC Exit */
#define KEY_MOVE		175
#define KEY_EDIT		176
#define KEY_SCROLLUP		177
#define KEY_SCROLLDOWN		178
#define KEY_KPLEFTPAREN		179
#define KEY_KPRIGHTPAREN	180
#define KEY_NEW			181	/* AC New */
#define KEY_REDO		182	/* AC Redo/Repeat */

#define KEY_F13			183
#define KEY_F14			184
#define KEY_F15			185
#define KEY_F16			186
#define KEY_F17			187
#define KEY_F18			188
#define KEY_F19			189
#define KEY_F20			190
#define KEY_F21			191
#define KEY_F22			192
#define KEY_F23			193
#define KEY_F24			194

#define KEY_PLAYCD		200
#define KEY_PAUSECD		201
#define KEY_PROG3		202
#define KEY_PROG4		203
#define KEY_DASHBOARD		204	/* AL Dashboard */
#define KEY_SUSPEND		205
#define KEY_CLOSE		206	/* AC Close */
#define KEY_PLAY		207
#define KEY_FASTFORWARD		208
#define KEY_BASSBOOST		209
#define KEY_PRINT		210	/* AC Print */
#define KEY_HP			211
#define KEY_CAMERA		212
#define KEY_SOUND		213
#define KEY_QUESTION		214
#define KEY_EMAIL		215
#define KEY_CHAT		216
#define KEY_SEARCH		217
#define KEY_CONNECT		218
#define KEY_FINANCE		219	/* AL Checkbook/Finance */
#define KEY_SPORT		220
#define KEY_SHOP		221
#define KEY_ALTERASE		222
#define KEY_CANCEL		223	/* AC Cancel */
#define KEY_BRIGHTNESSDOWN	224
#define KEY_BRIGHTNESSUP	225
#define KEY_MEDIA		226

#define KEY_SWITCHVIDEOMODE	227	/* Cycle between available video
					   outputs (Monitor/LCD/TV-out/etc) */
#define KEY_KBDILLUMTOGGLE	228
#define KEY_KBDILLUMDOWN	229
#define KEY_KBDILLUMUP		230

#define KEY_SEND		231	/* AC Send */
#define KEY_REPLY		232	/* AC Reply */
#define KEY_FORWARDMAIL		233	/* AC Forward Msg */
#define KEY_SAVE		234	/* AC Save */
#define KEY_DOCUMENTS		235

#define KEY_BATTERY		236

#define KEY_BLUETOOTH		237
#define KEY_WLAN		238
#define KEY_UWB			239

#define KEY_UNKNOWN		240

#define KEY_VIDEO_NEXT		241	/* drive next video source */
#define KEY_VIDEO_PREV		242	/* drive previous video source */
#define KEY_BRIGHTNESS_CYCLE	243	/* brightness up, after max is min */
#define KEY_BRIGHTNESS_ZERO	244	/* brightness off, use ambient */
#define KEY_DISPLAY_OFF		245	/* display device to off state */

#define KEY_WIMAX		246
#define KEY_RFKILL		247	/* Key that controls all radios */

/* Code 255 is reserved for special needs of AT keyboard driver */

#define BTN_MISC		0x100
#define BTN_0			0x100
#define BTN_1			0x101
#define BTN_2			0x102
#define BTN_3			0x103
#define BTN_4			0x104
#define BTN_5			0x105
#define BTN_6			0x106
#define BTN_7			0x107
#define BTN_8			0x108
#define BTN_9			0x109

#define BTN_MOUSE		0x110
#define BTN_LEFT		0x110
#define BTN_RIGHT		0x111
#define BTN_MIDDLE		0x112
#define BTN_SIDE		0x113
#define BTN_EXTRA		0x114
#define BTN_FORWARD		0x115
#define BTN_BACK		0x116
#define BTN_TASK		0x117

#define BTN_JOYSTICK		0x120
#define BTN_TRIGGER		0x120
#define BTN_THUMB		0x121
#define BTN_THUMB2		0x122
#define BTN_TOP			0x123
#define BTN_TOP2		0x124
#define BTN_PINKIE		0x125
#define BTN_BASE		0x126
#define BTN_BASE2		0x127
#define BTN_BASE3		0x128
#define BTN_BASE4		0x129
#define BTN_BASE5		0x12a
#define BTN_BASE6		0x12b
#define BTN_DEAD		0x12f

#define BTN_GAMEPAD		0x130
#define BTN_A			0x130
#define BTN_B			0x131
#define BTN_C			0x132
#define BTN_X			0x133
#define BTN_Y			0x134
#define BTN_Z			0x135
#define BTN_TL			0x136
#define BTN_TR			0x137
#define BTN_TL2			0x138
#define BTN_TR2			0x139
#define BTN_SELECT		0x13a
#define BTN_START		0x13b
#define BTN_MODE		0x13c
#define BTN_THUMBL		0x13d
#define BTN_THUMBR		0x13e

#define BTN_DIGI		0x140
#define BTN_TOOL_PEN		0x140
#define BTN_TOOL_RUBBER		0x141
#define BTN_TOOL_BRUSH		0x142
#define BTN_TOOL_PENCIL		0x143
#define BTN_TOOL_AIRBRUSH	0x144
#define BTN_TOOL_FINGER		0x145
#define BTN_TOOL_MOUSE		0x146
#define BTN_TOOL_LENS		0x147
#define BTN_TOOL_QUINTTAP	0x148	/* Five fingers on trackpad */
#define BTN_TOUCH		0x14a
#define BTN_STYLUS		0x14b
#define BTN_STYLUS2		0x14c
#define BTN_TOOL_DOUBLETAP	0x14d
#define BTN_TOOL_TRIPLETAP	0x14e
#define BTN_TOOL_QUADTAP	0x14f	/* Four fingers on trackpad */

#define BTN_WHEEL		0x150
#define BTN_GEAR_DOWN		0x150
#define BTN_GEAR_UP		0x151

#define KEY_OK			0x160
#define KEY_SELECT		0x161
#define KEY_GOTO		0x162
#define KEY_CLEAR		0x163
#define KEY_POWER2		0x164
#define KEY_OPTION		0x165
#define KEY_INFO		0x166	/* AL OEM Features/Tips/Tutorial */
#define KEY_TIME		0x167
#define KEY_VENDOR		0x168
#define KEY_ARCHIVE		0x169
#define KEY_PROGRAM		0x16a	/* Media Select Program Guide */
#define KEY_CHANNEL		0x16b
#define KEY_FAVORITES		0x16c
#define KEY_EPG			0x16d
#define KEY_PVR			0x16e	/* Media Select Home */
#define KEY_MHP			0x16f
#define KEY_LANGUAGE		0x170
#define KEY_TITLE		0x171
#define KEY_SUBTITLE		0x172
#define KEY_ANGLE		0x173
#define KEY_ZOOM		0x174
#define KEY_MODE		0x175
#define KEY_KEYBOARD		0x176
#define KEY_SCREEN		0x177
#define KEY_PC			0x178	/* Media Select Computer */
#define KEY_TV			0x179	/* Media Select TV */
#define KEY_TV2			0x17a	/* Media Select Cable */
#define KEY_VCR			0x17b	/* Media Select VCR */
#define KEY_VCR2		0x17c	/* VCR Plus */
#define KEY_SAT			0x17d	/* Media Select Satellite */
#define KEY_SAT2		0x17e
#define KEY_CD			0x17f	/* Media Select CD */
#define KEY_TAPE		0x180	/* Media Select Tape */
#define KEY_RADIO		0x181
#define KEY_TUNER		0x182	/* Media Select Tuner */
#define KEY_PLAYER		0x183
#define KEY_TEXT		0x184
#define KEY_DVD			0x185	/* Media Select DVD */
#define KEY_AUX			0x186
#define KEY_MP3			0x187
#define KEY_AUDIO		0x188	/* AL Audio Browser */
#define KEY_VIDEO		0x189	/* AL Movie Browser */
#define KEY_DIRECTORY		0x18a
#define KEY_LIST		0x18b
#define KEY_MEMO		0x18c	/* Media Select Messages */
#define KEY_CALENDAR		0x18d
#define KEY_RED			0x18e
#define KEY_GREEN		0x18f
#define KEY_YELLOW		0x190
#define KEY_BLUE		0x191
#define KEY_CHANNELUP		0x192	/* Channel Increment */
#define KEY_CHANNELDOWN		0x193	/* Channel Decrement */
#define KEY_FIRST		0x194
#define KEY_LAST		0x195	/* Recall Last */
#define KEY_AB			0x196
#define KEY_NEXT		0x197
#define KEY_RESTART		0x198
#define KEY_SLOW		0x199
#define KEY_SHUFFLE		0x19a
#define KEY_BREAK		0x19b
#define KEY_PREVIOUS		0x19c
#define KEY_DIGITS		0x19d
#define KEY_TEEN		0x19e
#define KEY_TWEN		0x19f
#define KEY_VIDEOPHONE		0x1a0	/* Media Select Video Phone */
#define KEY_GAMES		0x1a1	/* Media Select Games */
#define KEY_ZOOMIN		0x1a2	/* AC Zoom In */
#define KEY_ZOOMOUT		0x1a3	/* AC Zoom Out */
#define KEY_ZOOMRESET		0x1a4	/* AC Zoom */
#define KEY_WORDPROCESSOR	0x1a5	/* AL Word Processor */
#define KEY_EDITOR		0x1a6	/* AL Text Editor */
#define KEY_SPREADSHEET		0x1a7	/* AL Spreadsheet */
#define KEY_GRAPHICSEDITOR	0x1a8	/* AL Graphics Editor */
#define KEY_PRESENTATION	0x1a9	/* AL Presentation App */
#define KEY_DATABASE		0x1aa	/* AL Database App */
#define KEY_NEWS		0x1ab	/* AL Newsreader */
#define KEY_VOICEMAIL		0x1ac	/* AL Voicemail */
#define KEY_ADDRESSBOOK		0x1ad	/* AL Contacts/Address Book */
#define KEY_MESSENGER		0x1ae	/* AL Instant Messaging */
#define KEY_DISPLAYTOGGLE	0x1af	/* Turn display (LCD) on and off */
#define KEY_SPELLCHECK		0x1b0   /* AL Spell Check */
#define KEY_LOGOFF		0x1b1   /* AL Logoff */

#define KEY_DOLLAR		0x1b2
#define KEY_EURO		0x1b3

#define KEY_FRAMEBACK		0x1b4	/* Consumer - transport controls */
#define KEY_FRAMEFORWARD	0x1b5
#define KEY_CONTEXT_MENU	0x1b6	/* GenDesc - system context menu */
#define KEY_MEDIA_REPEAT	0x1b7	/* Consumer - transport control */
#define KEY_10CHANNELSUP	0x1b8	/* 10 channels up (10+) */
#define KEY_10CHANNELSDOWN	0x1b9	/* 10 channels down (10-) */
#define KEY_IMAGES		0x1ba	/* AL Image Browser */

#define KEY_DEL_EOL		0x1c0
#define KEY_DEL_EOS		0x1c1
#define KEY_INS_LINE		0x1c2
#define KEY_DEL_LINE		0x1c3

#define KEY_FN			0x1d0
#define KEY_FN_ESC		0x1d1
#define KEY_FN_F1		0x1d2
#define KEY_FN_F2		0x1d3
#define KEY_FN_F3		0x1d4
#define KEY_FN_F4		0x1d5
#define KEY_FN_F5		0x1d6
#define KEY_FN_F6		0x1d7
#define KEY_FN_F7		0x1d8
#define KEY_FN_F8		0x1d9
#define KEY_FN_F9		0x1da
#define KEY_FN_F10		0x1db
#define KEY_FN_F11		0x1dc
#define KEY_FN_F12		0x1dd
#define KEY_FN_1		0x1de
#define KEY_FN_2		0x1df
#define KEY_FN_D		0x1e0
#define KEY_FN_E		0x1e1
#define KEY_FN_F		0x1e2
#define KEY_FN_S		0x1e3
#define KEY_FN_B		0x1e4

#define KEY_BRL_DOT1		0x1f1
#define KEY_BRL_DOT2		0x1f2
#define KEY_BRL_DOT3		0x1f3
#define KEY_BRL_DOT4		0x1f4
#define KEY_BRL_DOT5		0x1f5
#define KEY_BRL_DOT6		0x1f6
#define KEY_BRL_DOT7		0x1f7
#define KEY_BRL_DOT8		0x1f8
#define KEY_BRL_DOT9		0x1f9
#define KEY_BRL_DOT10		0x1fa

#define KEY_NUMERIC_0		0x200	/* used by phones, remote controls, */
#define KEY_NUMERIC_1		0x201	/* and other keypads */
#define KEY_NUMERIC_2		0x202
#define KEY_NUMERIC_3		0x203
#define KEY_NUMERIC_4		0x204
#define KEY_NUMERIC_5		0x205
#define KEY_NUMERIC_6		0x206
#define KEY_NUMERIC_7		0x207
#define KEY_NUMERIC_8		0x208
#define KEY_NUMERIC_9		0x209
#define KEY_NUMERIC_STAR	0x20a
#define KEY_NUMERIC_POUND	0x20b

#define KEY_CAMERA_FOCUS	0x210
#define KEY_WPS_BUTTON		0x211	/* WiFi Protected Setup key */

#define KEY_TOUCHPAD_TOGGLE	0x212	/* Request switch touchpad on or off */
#define KEY_TOUCHPAD_ON		0x213
#define KEY_TOUCHPAD_OFF	0x214

#define KEY_CAMERA_ZOOMIN	0x215
#define KEY_CAMERA_ZOOMOUT	0x216
#define KEY_CAMERA_UP		0x217
#define KEY_CAMERA_DOWN		0x218
#define KEY_CAMERA_LEFT		0x219
#define KEY_CAMERA_RIGHT	0x21a

#define BTN_TRIGGER_HAPPY		0x2c0
#define BTN_TRIGGER_HAPPY1		0x2c0
#define BTN_TRIGGER_HAPPY2		0x2c1
#define BTN_TRIGGER_HAPPY3		0x2c2
#define BTN_TRIGGER_HAPPY4		0x2c3
#define BTN_TRIGGER_HAPPY5		0x2c4
#define BTN_TRIGGER_HAPPY6		0x2c5
#define BTN_TRIGGER_HAPPY7		0x2c6
#define BTN_TRIGGER_HAPPY8		0x2c7
#define BTN_TRIGGER_HAPPY9		0x2c8
#define BTN_TRIGGER_HAPPY10		0x2c9
#define BTN_TRIGGER_HAPPY11		0x2ca
#define BTN_TRIGGER_HAPPY12		0x2cb
#define BTN_TRIGGER_HAPPY13		0x2cc
#define BTN_TRIGGER_HAPPY14		0x2cd
#define BTN_TRIGGER_HAPPY15		0x2ce
#define BTN_TRIGGER_HAPPY16		0x2cf
#define BTN_TRIGGER_HAPPY17		0x2d0
#define BTN_TRIGGER_HAPPY18		0x2d1
#define BTN_TRIGGER_HAPPY19		0x2d2
#define BTN_TRIGGER_HAPPY20		0x2d3
#define BTN_TRIGGER_HAPPY21		0x2d4
#define BTN_TRIGGER_HAPPY22		0x2d5
#define BTN_TRIGGER_HAPPY23		0x2d6
#define BTN_TRIGGER_HAPPY24		0x2d7
#define BTN_TRIGGER_HAPPY25		0x2d8
#define BTN_TRIGGER_HAPPY26		0x2d9
#define BTN_TRIGGER_HAPPY27		0x2da
#define BTN_TRIGGER_HAPPY28		0x2db
#define BTN_TRIGGER_HAPPY29		0x2dc
#define BTN_TRIGGER_HAPPY30		0x2dd
#define BTN_TRIGGER_HAPPY31		0x2de
#define BTN_TRIGGER_HAPPY32		0x2df
#define BTN_TRIGGER_HAPPY33		0x2e0
#define BTN_TRIGGER_HAPPY34		0x2e1
#define BTN_TRIGGER_HAPPY35		0x2e2
#define BTN_TRIGGER_HAPPY36		0x2e3
#define BTN_TRIGGER_HAPPY37		0x2e4
#define BTN_TRIGGER_HAPPY38		0x2e5
#define BTN_TRIGGER_HAPPY39		0x2e6
#define BTN_TRIGGER_HAPPY40		0x2e7

/* We avoid low common keys in module aliases so they don't get huge. */
#define KEY_MIN_INTERESTING	KEY_MUTE
#define KEY_MAX			0x2ff
#define KEY_CNT			(KEY_MAX+1)

/*
 * Relative axes
 */

#define REL_X			0x00
#define REL_Y			0x01
#define REL_Z			0x02
#define REL_RX			0x03
#define REL_RY			0x04
#define REL_RZ			0x05
#define REL_HWHEEL		0x06
#define REL_DIAL		0x07
#define REL_WHEEL		0x08
#define REL_MISC		0x09
#define REL_MAX			0x0f
#define REL_CNT			(REL_MAX+1)

/*
 * Absolute axes
 */

#define ABS_X			0x00
#define ABS_Y			0x01
#define ABS_Z			0x02
#define ABS_RX			0x03
#define ABS_RY			0x04
#define ABS_RZ			0x05
#define ABS_THROTTLE		0x06
#define ABS_RUDDER		0x07
#define ABS_WHEEL		0x08
#define ABS_GAS			0x09
#define ABS_BRAKE		0x0a
#define ABS_HAT0X		0x10
#define ABS_HAT0Y		0x11
#define ABS_HAT1X		0x12
#define ABS_HAT1Y		0x13
#define ABS_HAT2X		0x14
#define ABS_HAT2Y		0x15
#define ABS_HAT3X		0x16
#define ABS_HAT3Y		0x17
#define ABS_PRESSURE		0x18
#define ABS_DISTANCE		0x19
#define ABS_TILT_X		0x1a
#define ABS_TILT_Y		0x1b
#define ABS_TOOL_WIDTH		0x1c

#define ABS_VOLUME		0x20

#define ABS_MISC		0x28

#define ABS_MT_SLOT		0x2f	/* MT slot being modified */
#define ABS_MT_TOUCH_MAJOR	0x30	/* Major axis of touching ellipse */
#define ABS_MT_TOUCH_MINOR	0x31	/* Minor axis (omit if circular) */
#define ABS_MT_WIDTH_MAJOR	0x32	/* Major axis of approaching ellipse */
#define ABS_MT_WIDTH_MINOR	0x33	/* Minor axis (omit if circular) */
#define ABS_MT_ORIENTATION	0x34	/* Ellipse orientation */
#define ABS_MT_POSITION_X	0x35	/* Center X ellipse position */
#define ABS_MT_POSITION_Y	0x36	/* Center Y ellipse position */
#define ABS_MT_TOOL_TYPE	0x37	/* Type of touching device */
#define ABS_MT_BLOB_ID		0x38	/* Group a set of packets as a blob */
#define ABS_MT_TRACKING_ID	0x39	/* Unique ID of initiated contact */
#define ABS_MT_PRESSURE		0x3a	/* Pressure on contact area */
#define ABS_MT_DISTANCE		0x3b	/* Contact hover distance */

#define ABS_MT_FIRST		ABS_MT_TOUCH_MAJOR
#define ABS_MT_LAST		ABS_MT_DISTANCE

#define ABS_MAX			0x3f
#define ABS_CNT			(ABS_MAX+1)

/*
 * Switch events
 */

#define SW_LID			0x00  /* set = lid shut */
#define SW_TABLET_MODE		0x01  /* set = tablet mode */
#define SW_HEADPHONE_INSERT	0x02  /* set = inserted */
#define SW_RFKILL_ALL		0x03  /* rfkill master switch, type "any"
					 set = radio enabled */
#define SW_RADIO		SW_RFKILL_ALL	/* deprecated */
#define SW_MICROPHONE_INSERT	0x04  /* set = inserted */
#define SW_DOCK			0x05  /* set = plugged into dock */
#define SW_LINEOUT_INSERT	0x06  /* set = inserted */
#define SW_JACK_PHYSICAL_INSERT 0x07  /* set = mechanical switch set */
#define SW_VIDEOOUT_INSERT	0x08  /* set = inserted */
#define SW_CAMERA_LENS_COVER	0x09  /* set = lens covered */
#define SW_KEYPAD_SLIDE		0x0a  /* set = keypad slide out */
#define SW_FRONT_PROXIMITY	0x0b  /* set = front proximity sensor active */
#define SW_ROTATE_LOCK		0x0c  /* set = rotate locked/disabled */
#define SW_MAX			0x0f
#define SW_CNT			(SW_MAX+1)

/*
 * Misc events
 */

#define MSC_SERIAL		0x00
#define MSC_PULSELED		0x01
#define MSC_GESTURE		0x02
#define MSC_RAW			0x03
#define MSC_SCAN		0x04
#define MSC_MAX			0x07
#define MSC_CNT			(MSC_MAX+1)

/*
 * LEDs
 */

#define LED_NUML		0x00
#define LED_CAPSL		0x01
#define LED_SCROLLL		0x02
#define LED_COMPOSE		0x03
#define LED_KANA		0x04
#define LED_SLEEP		0x05
#define LED_SUSPEND		0x06
#define LED_MUTE		0x07
#define LED_MISC		0x08
#define LED_MAIL		0x09
#define LED_CHARGING		0x0a
#define LED_MAX			0x0f
#define LED_CNT			(LED_MAX+1)

/*
 * Autorepeat values
 */

#define REP_DELAY		0x00
#define REP_PERIOD		0x01
#define REP_MAX			0x01
#define REP_CNT			(REP_MAX+1)

/*
 * Sounds
 */

#define SND_CLICK		0x00
#define SND_BELL		0x01
#define SND_TONE		0x02
#define SND_MAX			0x07
#define SND_CNT			(SND_MAX+1)

/*
 * IDs.
 */

#define ID_BUS			0
#define ID_VENDOR		1
#define ID_PRODUCT		2
#define ID_VERSION		3

#define BUS_PCI			0x01
#define BUS_ISAPNP		0x02
#define BUS_USB			0x03
#define BUS_HIL			0x04
#define BUS_BLUETOOTH		0x05
#define BUS_VIRTUAL		0x06

#define BUS_ISA			0x10
#define BUS_I8042		0x11
#define BUS_XTKBD		0x12
#define BUS_RS232		0x13
#define BUS_GAMEPORT		0x14
#define BUS_PARPORT		0x15
#define BUS_AMIGA		0x16
#define BUS_ADB			0x17
#define BUS_I2C			0x18
#define BUS_HOST		0x19
#define BUS_GSC			0x1A
#define BUS_ATARI		0x1B
#define BUS_SPI			0x1C

/*
 * MT_TOOL types
 */
#define MT_TOOL_FINGER		0
#define MT_TOOL_PEN		1
#define MT_TOOL_MAX		1

/*
 * Values describing the status of a force-feedback effect
 */
#define FF_STATUS_STOPPED	0x00
#define FF_STATUS_PLAYING	0x01
#define FF_STATUS_MAX		0x01

/*
 * Force feedback effect types
 */

#define FF_RUMBLE	0x50
#define FF_PERIODIC	0x51
#define FF_CONSTANT	0x52
#define FF_SPRING	0x53
#define FF_FRICTION	0x54
#define FF_DAMPER	0x55
#define FF_INERTIA	0x56
#define FF_RAMP		0x57

#define FF_EFFECT_MIN	FF_RUMBLE
#define FF_EFFECT_MAX	FF_RAMP

/*
 * Force feedback periodic effect types
 */

#define FF_SQUARE	0x58
#define FF_TRIANGLE	0x59
#define FF_SINE		0x5a
#define FF_SAW_UP	0x5b
#define FF_SAW_DOWN	0x5c
#define FF_CUSTOM	0x5d

#define FF_WAVEFORM_MIN	FF_SQUARE
#define FF_WAVEFORM_MAX	FF_CUSTOM

/*
 * Set ff device properties
 */

#define FF_GAIN		0x60
#define FF_AUTOCENTER	0x61

#define FF_MAX		0x7f
#define FF_CNT		(FF_MAX+1)

/**
 * struct input_dev - represents an input device
 * @head: list head for global input handlers list
 * @name: unique name of the device
 * @phys: physical path to the device in the system hierarchy
 * @uniq: unique identification code for the device (if device has it)
 * @id: id of the device (struct input_id)
 * @propbit: bitmap of device properties and quirks
 * @evbit: bitmap of types of events supported by the device (EV_KEY,
 *	EV_REL, etc.)
 * @keybit: bitmap of keys/buttons this device has
 * @relbit: bitmap of relative axes for the device
 * @absbit: bitmap of absolute axes for the device
 * @mscbit: bitmap of miscellaneous events supported by the device
 * @ledbit: bitmap of leds present on the device
 * @sndbit: bitmap of sound effects supported by the device
 * @ffbit: bitmap of force feedback effects supported by the device
 * @swbit: bitmap of switches present on the device
 * @hint_events_per_packet: average number of events generated by the
 *	device in a packet (between EV_SYN/SYN_REPORT events). Used by
 *	event handlers to estimate size of the buffer needed to hold
 *	events.
 * @keycodemax: size of keycode table
 * @keycodesize: size of elements in keycode table
 * @keycode: map of scancodes to keycodes for this device
 * @getkeycode: optional legacy method to retrieve current keymap.
 * @setkeycode: optional method to alter current keymap, used to implement
 *	sparse keymaps. If not supplied default mechanism will be used.
 *	The method is being called while holding event_lock and thus must
 *	not sleep
 * @repeat_key: stores key code of the last key pressed; used to implement
 *	software autorepeat
 * @repeat_ev: timer event for software autorepeat
 * @rep: current values for autorepeat parameters (delay, rate)
 * @mt: pointer to array of struct input_mt_slot holding current values
 *	of tracked contacts
 * @mtsize: number of MT slots the device uses
 * @slot: MT slot currently being transmitted
 * @trkid: stores MT tracking ID for the current contact
 * @absinfo: array of &struct input_absinfo elements holding information
 *	about absolute axes (current value, min, max, flat, fuzz,
 *	resolution)
 * @key: reflects current state of device's keys/buttons
 * @led: reflects current state of device's LEDs
 * @snd: reflects current state of sound effects
 * @sw: reflects current state of device's switches
 * @open: this method is called when the very first user calls
 *	input_open_device(). The driver must prepare the device
 *	to start generating events (start polling thread,
 *	request an IRQ, submit URB, etc.)
 * @close: this method is called when the very last user calls
 *	input_close_device().
 * @flush: purges the device. Most commonly used to get rid of force
 *	feedback effects loaded into the device when disconnecting
 *	from it
 * @event: event handler for events sent _to_ the device, like EV_LED
 *	or EV_SND. The device is expected to carry out the requested
 *	action (turn on a LED, play sound, etc.) The call is protected
 *	by @event_lock and must not sleep
 * @event_lock: this spinlock is is taken when input core receives
 *	and processes a new event for the device (in input_event()).
 *	Code that accesses and/or modifies parameters of a device
 *	(such as keymap or absmin, absmax, absfuzz, etc.) after device
 *	has been registered with input core must take this lock.
 * @ops_lock: this spinlock is is taken when doing input device operation
 * @users: stores number of users (input handlers). It is used to make sure
 *	that dev->open() is only called when first input handler connects
 *	and dev->close() is called when very last input handler disconnects
 * @going_away: marks devices that are in a middle of unregistering
 * @sync: set to %true when there were no new events since last EV_SYN
 * @dev: driver model's view of this device
 * @priv: driver specific data
 */
struct input_dev {
	struct dlist head;

	const char *name;
	const char *phys;
	const char *uniq;
	struct input_id id;

	unsigned long propbit[BITS_TO_LONGS(INPUT_PROP_CNT)];

	unsigned long evbit[BITS_TO_LONGS(EV_CNT)];
	unsigned long keybit[BITS_TO_LONGS(KEY_CNT)];
	unsigned long relbit[BITS_TO_LONGS(REL_CNT)];
	unsigned long absbit[BITS_TO_LONGS(ABS_CNT)];
	unsigned long mscbit[BITS_TO_LONGS(MSC_CNT)];
	unsigned long ledbit[BITS_TO_LONGS(LED_CNT)];
	unsigned long sndbit[BITS_TO_LONGS(SND_CNT)];
	unsigned long ffbit[BITS_TO_LONGS(FF_CNT)];
	unsigned long swbit[BITS_TO_LONGS(SW_CNT)];

	unsigned int hint_events_per_packet;

	unsigned int keycodemax;
	unsigned int keycodesize;
	void *keycode;

	int (*setkeycode)(struct input_dev *dev,
			  const struct input_keymap_entry *ke,
			  unsigned int *old_keycode);
	int (*getkeycode)(struct input_dev *dev,
			  struct input_keymap_entry *ke);

	unsigned int repeat_key;
	struct vmm_timer_event repeat_ev;

	int rep[REP_CNT];

	struct input_mt *mt;
	struct input_absinfo *absinfo;

	unsigned long key[BITS_TO_LONGS(KEY_CNT)];
	unsigned long led[BITS_TO_LONGS(LED_CNT)];
	unsigned long snd[BITS_TO_LONGS(SND_CNT)];
	unsigned long sw[BITS_TO_LONGS(SW_CNT)];

	int (*open)(struct input_dev *dev);
	void (*close)(struct input_dev *dev);
	int (*flush)(struct input_dev *dev);
	int (*event)(struct input_dev *dev,
		     unsigned int type, unsigned int code, int value);

	vmm_spinlock_t event_lock;
	vmm_spinlock_t ops_lock;

	unsigned int users;
	bool going_away;
	bool sync;

	struct vmm_device dev;

	void *priv;
};

/**
 * struct input_handler - represents an input handler
 * @head: list head for global input handlers list
 * @connected: flag showing whether input handler is connected to events
 * @conn_head: list heads for connected input handlers list
 * @name: unique name of the input handler
 * @evbit: bitmap of types of events supported by the device (EV_KEY,
 *	EV_REL, etc.)
 * @event: callback function of input handler to handle events
 * @priv: handler specific data
 */
struct input_handler {
	struct dlist head;

	bool connected;
	struct dlist conn_head[EV_CNT];

	const char *name;

	unsigned long evbit[BITS_TO_LONGS(EV_CNT)];

	int (*event)(struct input_handler *ihnd,
		     struct input_dev *dev,
		     unsigned int type, unsigned int code, int value);

	void *priv;
};

static inline void *input_get_drvdata(struct input_dev *dev)
{
	return dev->priv;
}

static inline void input_set_drvdata(struct input_dev *dev, void *data)
{
	dev->priv = data;
}

/** Allocate an input device
 *
 * Returns prepared struct input_dev or NULL.
 */
struct input_dev *input_allocate_device(void);

/** Free an input device
 * @dev: input device to free
 */
void input_free_device(struct input_dev *dev);

/** Register input device
 * @dev: device to be registered
 *
 * This function registers device with input core. The device must be
 * allocated with input_allocate_device() and all it's capabilities
 * set up before registering.
 */
int input_register_device(struct input_dev *dev);

/** Unregister input device
 * @dev: device to be unregistered
 *
 * This function unregisters an input device. Once device is unregistered
 * the caller should not try to access it as it may get freed at any moment.
 */
int input_unregister_device(struct input_dev *dev);

/** Reset/restore input device
 * @dev: input device whose state needs to be reset
 *
 * This function tries to reset the state of an opened input device and
 * bring internal state and state if the hardware in sync with each other.
 * We mark all keys as released, restore LED state, repeat rate, etc.
 */
void input_reset_device(struct input_dev *dev);

/** Flush input device
 * @dev: input device whose state needs to be reset
 *
 * This function tries to flush an input device.
 */
int input_flush_device(struct input_dev *dev);

/** Find a input device based on its physical name */
struct input_dev *input_find_device(const char *phys);

/** Iterate over each input device */
int input_iterate_device(struct input_dev *start, void *data,
			 int (*fn)(struct input_dev *dev, void *data));

/** Count number of input devices */
u32 input_count_device(void);

/** Register input handler */
int input_register_handler(struct input_handler *ihnd);

/** Unregister input handler */
int input_unregister_handler(struct input_handler *ihnd);

/** Connect input handler to start receiving events */
int input_connect_handler(struct input_handler *ihnd);

/** Disconnect input handler to stop receiving events */
int input_disconnect_handler(struct input_handler *ihnd);

/** Find a input handler */
struct input_handler *input_find_handler(const char *name);

/** Get input handler with given index */
struct input_handler *input_get_handler(int index);

/** Count number of input handler */
u32 input_count_handler(void);

/** Report new input event
 * @dev: device that generated the event
 * @type: type of the event
 * @code: event code
 * @value: value of the event
 *
 * This function should be used by drivers implementing various input
 * devices to report input events.
 */
void input_event(struct input_dev *dev,
		 unsigned int type, unsigned int code, int value);

/** Report EV_KEY from input device */
static inline void input_report_key(struct input_dev *dev,
				    unsigned int code, int value)
{
	input_event(dev, EV_KEY, code, !!value);
}

/** Report EV_REL from input device */
static inline void input_report_rel(struct input_dev *dev,
				    unsigned int code, int value)
{
	input_event(dev, EV_REL, code, value);
}

/** Report EV_ABS from input device */
static inline void input_report_abs(struct input_dev *dev,
				    unsigned int code, int value)
{
	input_event(dev, EV_ABS, code, value);
}

/** Report EV_FF_STATUS from input device */
static inline void input_report_ff_status(struct input_dev *dev,
					  unsigned int code, int value)
{
	input_event(dev, EV_FF_STATUS, code, value);
}

/** Report EV_SW from input device */
static inline void input_report_switch(struct input_dev *dev,
				       unsigned int code, int value)
{
	input_event(dev, EV_SW, code, !!value);
}

/** Report SYNC event */
static inline void input_sync(struct input_dev *dev)
{
	input_event(dev, EV_SYN, SYN_REPORT, 0);
}

/** Report MultiTouch SYNC event */
static inline void input_mt_sync(struct input_dev *dev)
{
	input_event(dev, EV_SYN, SYN_MT_REPORT, 0);
}

/** Set capability of input device to generate event of give type and code
 * @dev: device that is capable of emitting or accepting event
 * @type: type of the event (EV_KEY, EV_REL, etc...)
 * @code: event code
 *
 * In addition to setting up corresponding bit in appropriate capability
 * bitmap the function also adjusts dev->evbit.
 */
void input_set_capability(struct input_dev *dev,
			  unsigned int type, unsigned int code);

/**
 * Tell handlers about the driver event rate
 * @dev: the input device used by the driver
 * @n_events: the average number of events between calls to input_sync()
 *
 * If the event rate sent from a device is unusually large, use this
 * function to set the expected event rate. This will allow handlers
 * to set up an appropriate buffer size for the event stream, in order
 * to minimize information loss.
 */
static inline void input_set_events_per_packet(struct input_dev *dev,
					       int n_events)
{
	dev->hint_events_per_packet = n_events;
}

/**
 * Allocates array of input_absinfo structs
 * @dev: the input device emitting absolute events
 *
 * If the absinfo struct the caller asked for is already allocated, this
 * functions will not do anything.
 */
void input_alloc_absinfo(struct input_dev *dev);

void input_set_abs_params(struct input_dev *dev, unsigned int axis,
			  int min, int max, int fuzz, int flat);

#define INPUT_GENERATE_ABS_ACCESSORS(_suffix, _item)			\
static inline int input_abs_get_##_suffix(struct input_dev *dev,	\
					  unsigned int axis)		\
{									\
	return dev->absinfo ? dev->absinfo[axis]._item : 0;		\
}									\
									\
static inline void input_abs_set_##_suffix(struct input_dev *dev,	\
					   unsigned int axis, int val)	\
{									\
	input_alloc_absinfo(dev);					\
	if (dev->absinfo)						\
		dev->absinfo[axis]._item = val;			\
}

INPUT_GENERATE_ABS_ACCESSORS(val, value)
INPUT_GENERATE_ABS_ACCESSORS(min, minimum)
INPUT_GENERATE_ABS_ACCESSORS(max, maximum)
INPUT_GENERATE_ABS_ACCESSORS(fuzz, fuzz)
INPUT_GENERATE_ABS_ACCESSORS(flat, flat)
INPUT_GENERATE_ABS_ACCESSORS(res, resolution)

/** Convert input keymap scancode to scalar value
 * @ke: keymap entry containing scancode to be converted.
 * @scancode: pointer to the location where converted scancode should
 *	be stored.
 *
 * This function is used to convert scancode stored in &struct keymap_entry
 * into scalar form understood by legacy keymap handling methods. These
 * methods expect scancodes to be represented as 'unsigned int'.
 */
int input_scancode_to_scalar(const struct input_keymap_entry *ke,
				 unsigned int *scancode);

/** Retrieve keycode currently mapped to a given scancode
 * @dev: input device which keymap is being queried
 * @ke: keymap entry
 *
 * This function should be called by anyone interested in retrieving current
 * keymap. Presently evdev handlers use it.
 */
int input_get_keycode(struct input_dev *dev,
			  struct input_keymap_entry *ke);

/** Attribute a keycode to a given scancode
 * @dev: input device which keymap is being updated
 * @ke: new keymap entry
 *
 * This function should be called by anyone needing to update current
 * keymap. Presently keyboard and evdev handlers use it.
 */
int input_set_keycode(struct input_dev *dev,
			  const struct input_keymap_entry *ke);

#endif /* __DRV_INPUT_H_ */
