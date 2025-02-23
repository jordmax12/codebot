const dbUtils = require('../helpers/dbUtils');

class UserController {
  static async getUserById(id) {
    await dbUtils.connect();
    const result = await dbUtils.query('SELECT * FROM users WHERE id = ?', [id]);
    return result[0] || null;
  }

  static async createUser(userData) {
    await dbUtils.connect();
    const result = await dbUtils.query(
      'INSERT INTO users (name, email) VALUES (?, ?)',
      [userData.name, userData.email]
    );
    return { id: result.insertId, ...userData };
  }
}

module.exports = UserController;