/*
Navicat MySQL Data Transfer

Source Server         : localhost_3306
Source Server Version : 50711
Source Host           : localhost:3306
Source Database       : sharpv

Target Server Type    : MYSQL
Target Server Version : 50711
File Encoding         : 65001

Date: 2017-04-04 17:01:00
*/

SET FOREIGN_KEY_CHECKS=0;
USE sharpv;
-- ----------------------------
-- Table structure for sharpv_email_verification
-- ----------------------------
DROP TABLE IF EXISTS `sharpv_email_verification`;
CREATE TABLE `sharpv_email_verification` (
  `email` varchar(64) COLLATE utf8mb4_unicode_ci NOT NULL,
  `token` varchar(36) COLLATE utf8mb4_unicode_ci NOT NULL,
  `expire_time` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`email`),
  UNIQUE KEY `token` (`token`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ----------------------------
-- Table structure for sharpv_users
-- ----------------------------
DROP TABLE IF EXISTS `sharpv_users`;
CREATE TABLE `sharpv_users` (
  `user_id` bigint(20) NOT NULL AUTO_INCREMENT,
  `username` varchar(32) COLLATE utf8mb4_unicode_ci NOT NULL,
  `password` varchar(32) COLLATE utf8mb4_unicode_ci NOT NULL,
  `email` varchar(64) COLLATE utf8mb4_unicode_ci NOT NULL,
  `user_group_id` int(4) NOT NULL,
  PRIMARY KEY (`user_id`),
  UNIQUE KEY `username` (`username`),
  UNIQUE KEY `email` (`email`),
  KEY `user_group_id` (`user_group_id`),
  CONSTRAINT `sharpv_users_ibfk_1` FOREIGN KEY (`user_group_id`) REFERENCES `sharpv_user_groups` (`user_group_id`)
) ENGINE=InnoDB AUTO_INCREMENT=4 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ----------------------------
-- Records of sharpv_users
-- ----------------------------
INSERT INTO `sharpv_users` VALUES ('0', 'Guest', '', 'guest@sharp-v.org', '2');
INSERT INTO `sharpv_users` VALUES ('1', 'root', 'e118b111376cffc0fcfb10e9dc43884b', 'webmaster@sharp-v.org', '3');
INSERT INTO `sharpv_users` VALUES ('2', 'hzxie', '785ee107c11dfe36de668b1ae7baacbb', 'cshzxie@gmail.com', '2');
INSERT INTO `sharpv_users` VALUES ('3', 'sunce', 'e10adc3949ba59abbe56e057f20f883e', 'spatriotc@gmail.com', '2');

-- ----------------------------
-- Table structure for sharpv_user_groups
-- ----------------------------
DROP TABLE IF EXISTS `sharpv_user_groups`;
CREATE TABLE `sharpv_user_groups` (
  `user_group_id` int(4) NOT NULL AUTO_INCREMENT,
  `user_group_slug` varchar(32) COLLATE utf8mb4_unicode_ci NOT NULL,
  `user_group_name` varchar(32) COLLATE utf8mb4_unicode_ci NOT NULL,
  PRIMARY KEY (`user_group_id`)
) ENGINE=InnoDB AUTO_INCREMENT=4 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ----------------------------
-- Records of sharpv_user_groups
-- ----------------------------
INSERT INTO `sharpv_user_groups` VALUES ('1', 'forbidden', 'Forbidden');
INSERT INTO `sharpv_user_groups` VALUES ('2', 'users', 'Users');
INSERT INTO `sharpv_user_groups` VALUES ('3', 'administrators', 'Administrators');
