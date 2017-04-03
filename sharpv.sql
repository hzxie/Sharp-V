-- phpMyAdmin SQL Dump
-- version 4.5.2
-- http://www.phpmyadmin.net
--
-- Host: localhost
-- Generation Time: Apr 01, 2017 at 07:03 AM
-- Server version: 10.1.19-MariaDB
-- PHP Version: 7.0.9

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `sharpv`
--

-- --------------------------------------------------------

--
-- Table structure for table `sharpv_users`
--

CREATE TABLE `sharpv_users` (
  `user_id` bigint(20) NOT NULL,
  `username` varchar(32) NOT NULL,
  `password` varchar(32) NOT NULL,
  `email` varchar(64) NOT NULL,
  `user_group_id` int(4) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `sharpv_users`
--

INSERT INTO `sharpv_users` (`user_id`, `username`, `password`, `email`, `user_group_id`) VALUES
(1, 'root', 'e118b111376cffc0fcfb10e9dc43884b', 'webmaster@sharp-v.org', 3),
(2, 'hzxie', '785ee107c11dfe36de668b1ae7baacbb', 'cshzxie@gmail.com', 2);

-- --------------------------------------------------------

--
-- Table structure for table `sharpv_user_groups`
--

CREATE TABLE `sharpv_user_groups` (
  `user_group_id` int(4) NOT NULL,
  `user_group_slug` varchar(32) NOT NULL,
  `user_group_name` varchar(32) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `sharpv_user_groups`
--

INSERT INTO `sharpv_user_groups` (`user_group_id`, `user_group_slug`, `user_group_name`) VALUES
(1, 'forbidden', 'Forbidden'),
(2, 'users', 'Users'),
(3, 'administrators', 'Administrators');

--
-- Indexes for dumped tables
--

--
-- Indexes for table `sharpv_users`
--
ALTER TABLE `sharpv_users`
  ADD PRIMARY KEY (`user_id`),
  ADD UNIQUE KEY `username` (`username`),
  ADD UNIQUE KEY `email` (`email`),
  ADD KEY `user_group_id` (`user_group_id`);

--
-- Indexes for table `sharpv_user_groups`
--
ALTER TABLE `sharpv_user_groups`
  ADD PRIMARY KEY (`user_group_id`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `sharpv_users`
--
ALTER TABLE `sharpv_users`
  MODIFY `user_id` bigint(20) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=3;
--
-- AUTO_INCREMENT for table `sharpv_user_groups`
--
ALTER TABLE `sharpv_user_groups`
  MODIFY `user_group_id` int(4) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=4;
--
-- Constraints for dumped tables
--

--
-- Constraints for table `sharpv_users`
--
ALTER TABLE `sharpv_users`
  ADD CONSTRAINT `sharpv_users_ibfk_1` FOREIGN KEY (`user_group_id`) REFERENCES `sharpv_user_groups` (`user_group_id`);

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
