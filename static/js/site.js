/*!
 * Global JavaScript by Haozhe Xie.
 *
 * Copyright 2015 Contributors
 * Released under the GPL v3 license
 * http://opensource.org/licenses/GPL-3.0
 */
String.prototype.format = function() {
    var newStr = this, 
        i = 0;
    while (/%s/.test(newStr)) {
        newStr = newStr.replace("%s", arguments[i++]);
    }
    return newStr;
}