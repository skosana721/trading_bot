// Vue.js Admin Portal Utilities
// This file provides shared utilities and components for Vue.js admin pages

// Global Vue Admin utilities
window.VueAdminUtils = {
    // Show alert message
    showAlert: function(message, type = 'info', duration = 5000) {
        // This will be handled by the Vue app's alert system
        // For compatibility with existing code
        if (window.currentVueApp && window.currentVueApp.showAlert) {
            window.currentVueApp.showAlert(message, type, duration);
        } else {
            // Fallback to browser alert
            alert(message);
        }
    },
    
    // Show loading spinner
    showLoading: function(element) {
        if (typeof element === 'string') {
            element = document.getElementById(element);
        }
        if (element) {
            element.innerHTML = `
                <div class="text-center">
                    <div class="spinner-border" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                </div>
            `;
        }
    },
    
    // Hide loading spinner
    hideLoading: function(element) {
        if (typeof element === 'string') {
            element = document.getElementById(element);
        }
        if (element) {
            const spinner = element.querySelector('.spinner-border');
            if (spinner) {
                spinner.remove();
            }
        }
    },
    
    // Format bytes
    formatBytes: function(bytes, decimals = 2) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const dm = decimals < 0 ? 0 : decimals;
        const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
    },
    
    // Format percentage
    formatPercentage: function(value, decimals = 1) {
        return value.toFixed(decimals) + '%';
    },
    
    // Format currency
    formatCurrency: function(amount, currency = 'USD') {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: currency
        }).format(amount);
    },
    
    // Format date
    formatDate: function(dateString) {
        if (!dateString) return 'N/A';
        const date = new Date(dateString);
        return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
    },
    
    // Format date only
    formatDateOnly: function(dateString) {
        if (!dateString) return 'N/A';
        const date = new Date(dateString);
        return date.toLocaleDateString();
    },
    
    // Format time only
    formatTimeOnly: function(dateString) {
        if (!dateString) return 'N/A';
        const date = new Date(dateString);
        return date.toLocaleTimeString();
    },
    
    // Confirm action
    confirmAction: function(message, callback) {
        if (confirm(message)) {
            callback();
        }
    },
    
    // Get admin API key from session
    getAdminApiKey: function() {
        // This would typically get the API key from a secure source
        // For now, we'll use a placeholder
        return 'admin-api-key';
    },
    
    // Create Vue app with common configuration
    createVueApp: function(config) {
        const app = Vue.createApp({
            data() {
                return {
                    alerts: [],
                    loading: false,
                    ...config.data
                }
            },
            mounted() {
                // Initialize axios defaults
                axios.defaults.headers.common['X-Requested-With'] = 'XMLHttpRequest';
                
                // Add response interceptor for error handling
                axios.interceptors.response.use(
                    response => response,
                    error => {
                        if (error.response && error.response.status === 401) {
                            this.showAlert('Session expired. Please login again.', 'warning');
                            setTimeout(() => {
                                window.location.href = '/admin/login';
                            }, 2000);
                        }
                        return Promise.reject(error);
                    }
                );
                
                // Store reference for global access
                window.currentVueApp = this;
                
                // Call custom mounted function if provided
                if (config.mounted) {
                    config.mounted.call(this);
                }
            },
            methods: {
                showAlert(message, type = 'info', duration = 5000) {
                    const alert = {
                        id: Date.now() + Math.random(),
                        message: message,
                        type: type
                    };
                    
                    this.alerts.push(alert);
                    
                    // Auto-remove alert after duration
                    if (duration > 0) {
                        setTimeout(() => {
                            this.removeAlert(alert.id);
                        }, duration);
                    }
                },
                
                removeAlert(alertId) {
                    this.alerts = this.alerts.filter(alert => alert.id !== alertId);
                },
                
                alertIcon(type) {
                    const icons = {
                        success: 'fas fa-check-circle',
                        error: 'fas fa-exclamation-circle',
                        warning: 'fas fa-exclamation-triangle',
                        info: 'fas fa-info-circle'
                    };
                    return icons[type] || icons.info;
                },
                
                formatCurrency(amount, currency = 'USD') {
                    return VueAdminUtils.formatCurrency(amount, currency);
                },
                
                formatPercentage(value) {
                    return VueAdminUtils.formatPercentage(value);
                },
                
                formatDate(dateString) {
                    return VueAdminUtils.formatDate(dateString);
                },
                
                formatDateOnly(dateString) {
                    return VueAdminUtils.formatDateOnly(dateString);
                },
                
                formatTimeOnly(dateString) {
                    return VueAdminUtils.formatTimeOnly(dateString);
                },
                
                getAdminApiKey() {
                    return VueAdminUtils.getAdminApiKey();
                },
                
                async logout() {
                    if (confirm('Are you sure you want to logout?')) {
                        try {
                            await axios.post('/admin/logout');
                            window.location.href = '/admin/login';
                        } catch (error) {
                            this.showAlert('Logout failed: ' + error.message, 'error');
                        }
                    }
                },
                
                // Merge custom methods
                ...config.methods
            }
        });
        
        return app;
    },
    
    // Common Vue components
    components: {
        // Alert component
        Alert: {
            props: ['alert'],
            template: `
                <div :class="'alert alert-' + alert.type + ' alert-dismissible fade show'" 
                     style="animation: slideIn 0.3s ease;">
                    <i :class="alertIcon(alert.type)"></i> {{ alert.message }}
                    <button type="button" class="btn-close" @click="$emit('remove', alert.id)"></button>
                </div>
            `,
            methods: {
                alertIcon(type) {
                    const icons = {
                        success: 'fas fa-check-circle',
                        error: 'fas fa-exclamation-circle',
                        warning: 'fas fa-exclamation-triangle',
                        info: 'fas fa-info-circle'
                    };
                    return icons[type] || icons.info;
                }
            }
        },
        
        // Loading spinner component
        LoadingSpinner: {
            props: {
                size: {
                    type: String,
                    default: 'normal'
                }
            },
            template: `
                <div class="text-center">
                    <div :class="'spinner-border ' + (size === 'sm' ? 'spinner-border-sm' : '')" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                </div>
            `
        },
        
        // Pagination component
        Pagination: {
            props: ['pagination'],
            template: `
                <nav v-if="pagination.total_pages > 1">
                    <ul class="pagination justify-content-center">
                        <li :class="'page-item ' + (!pagination.has_prev ? 'disabled' : '')">
                            <a class="page-link" href="#" @click.prevent="$emit('page-change', pagination.prev_page || 1)">Previous</a>
                        </li>
                        <li v-for="page in getPageNumbers()" :key="page" :class="'page-item ' + (page === pagination.current_page ? 'active' : '')">
                            <a class="page-link" href="#" @click.prevent="$emit('page-change', page)">{{ page }}</a>
                        </li>
                        <li :class="'page-item ' + (!pagination.has_next ? 'disabled' : '')">
                            <a class="page-link" href="#" @click.prevent="$emit('page-change', pagination.next_page || pagination.total_pages)">Next</a>
                        </li>
                    </ul>
                </nav>
            `,
            methods: {
                getPageNumbers() {
                    const pages = [];
                    const start = Math.max(1, this.pagination.current_page - 2);
                    const end = Math.min(this.pagination.total_pages, this.pagination.current_page + 2);
                    
                    for (let i = start; i <= end; i++) {
                        pages.push(i);
                    }
                    
                    return pages;
                }
            }
        }
    }
};

// Axios interceptors for admin requests
axios.interceptors.request.use(
    function(config) {
        // Add admin API key to requests
        if (config.url.includes('/admin/') && config.method !== 'get') {
            config.headers['X-Admin-API-Key'] = VueAdminUtils.getAdminApiKey();
        }
        return config;
    },
    function(error) {
        return Promise.reject(error);
    }
);

// Global logout function for compatibility
window.logout = function() {
    if (window.currentVueApp) {
        window.currentVueApp.logout();
    } else {
        if (confirm('Are you sure you want to logout?')) {
            axios.post('/admin/logout')
                .then(() => {
                    window.location.href = '/admin/login';
                })
                .catch(() => {
                    window.location.href = '/admin/login';
                });
        }
    }
};

// Initialize admin portal when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log('Vue Admin Portal utilities loaded');
    
    // Auto-refresh dashboard data every 30 seconds if on dashboard page
    if (window.location.pathname === '/admin/dashboard') {
        setInterval(function() {
            if (window.currentVueApp && typeof window.currentVueApp.loadDashboardData === 'function') {
                window.currentVueApp.loadDashboardData();
            }
        }, 30000);
    }
});

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = VueAdminUtils;
}
